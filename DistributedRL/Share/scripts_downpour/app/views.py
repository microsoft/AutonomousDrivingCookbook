from django.shortcuts import render
from django.http import HttpRequest, JsonResponse
from django.template import RequestContext
from django.views.decorators.csrf import csrf_exempt
from datetime import datetime
from ipware.ip import get_ip
import json
import inspect
import threading
import glob
import os
import sys
from app.rl_model import RlModel

# The main code file for the trainer.

# Initialize the RL model
if ('weights_path' in os.environ):
    rl_model = RlModel(os.environ['weights_path'], os.environ['train_conv_layers'].lower() == 'true')
else:
    rl_model = RlModel(None, os.environ['train_conv_layers'].lower() == 'true')
    
model_lock = threading.Lock()
batch_count = 0
batch_update_frequency = 0
next_batch_update_count = 0
checkpoint_dir = ''
agents_having_latest_critic = []

min_epsilon = float(os.environ['min_epsilon'])
epsilon_step = float(os.environ['per_iter_epsilon_reduction'])
epsilon = 1.0

# A simple endpoint that can be used to determine if the trainer is online.
# All requests will be responded to with a JSON {"message": "PONG"}
# Routed to /ping
@csrf_exempt
def ping(request):
    try:
        print('PONG')
        return JsonResponse({'message': 'pong'})
    finally:
        sys.stdout.flush()
        sys.stderr.flush()

# This endpoint is used to send gradient updates.
# It expects a POST request with the gradients in the body.
# It will return the latest model in the body
# Routed to /gradient_update
@csrf_exempt
def gradient_update(request):
    global rl_model
    global batch_count
    global agents_having_latest_critic
    global next_batch_update_count
    global batch_update_frequency
    global checkpoint_dir
    global agents_having_latest_critic
    global epsilon
    global epsilon_step
    global min_epsilon
    try:
        # Check that the request is a POST
        if (request.method != 'POST'):
            raise ValueError('Need post method, got {0}'.format(request.method))

        # Read in the data and determine which trainer sent the information
        post_data = json.loads(request.body.decode('utf-8'))
        request_ip = get_ip(request)

        print('request_ip is {0}'.format(request_ip))

        # Django does not play nicely with TensorFlow in a multi-threaded context.
        # Ensure that only a single minibatch is being processed at a time.
        # Other threads will enter a queue and will be processed once the lock is released.
        with model_lock:
            
            # Update the number of batches received
            batch_count += int(post_data['batch_count'])
            print('Received {0} batches. batch count is now {1}.'.format(int(post_data['batch_count']), batch_count))

            # We only occasionally update the critic (target) model. Determine if it's time to update the critic.
            should_update_critic = (batch_count >= next_batch_update_count)

            if (should_update_critic):
                print('updating critic this iter.')
            else:
                print('not updating critic')

            # Read in the gradients and update the model
            model_gradients = post_data['gradients']
            rl_model.update_with_gradient(model_gradients, should_update_critic)
            
            # If we updated the critic, checkpoint the model.
            if should_update_critic:
                print('checkpointing...')
                checkpoint_state()
                next_batch_update_count += batch_update_frequency
                agents_having_latest_critic = []

            # To save network bandwidth, we only need to send the critic if it's changed.
            # Create the response to send to the agent
            if request_ip not in agents_having_latest_critic:
                print('Agent {0} has not received the latest critic model. Sending both.'.format(request_ip))
                model_response = rl_model.to_packet(get_target=True)
                agents_having_latest_critic.append(request_ip)
            else:
                print('Agent {0} has received the latest critic model. Sending only the actor.'.format(request_ip))
                model_response = rl_model.to_packet(get_target=False)

            epsilon -= epsilon_step
            epsilon = max(epsilon, min_epsilon)
            
            print('Sending epsilon of {0} to {1}'.format(epsilon, request_ip))
            
            model_response['epsilon'] = epsilon
                
            # Send the response to the agent.
            return JsonResponse(model_response)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()

# An endpoint to get the latest model.
# It is expected to be called with a GET request.
# The response will be the model.
# Routed to /latest
@csrf_exempt
def get_latest_model(request):
    global rl_model
    try:
        if (request.method != 'GET'):
            raise ValueError('Need get method, got {0}'.format(request.method))

        with model_lock:
            model_response = rl_model.to_packet(get_target=True)
            return JsonResponse(model_response)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()

# A helper function to checkpoint the current state of the model.
@csrf_exempt
def checkpoint_state():
    global rl_model
    global batch_count
    try:
        checkpoint = {}
        checkpoint['model'] = rl_model.to_packet(get_target=True)
        checkpoint['batch_count'] = batch_count
        checkpoint_str = json.dumps(checkpoint)

        file_name = os.path.join(checkpoint_dir, '{0}.json'.format(batch_count)) 
        with open(file_name, 'w') as f:
            print('Checkpointing to {0}'.format(file_name))
            f.write(checkpoint_str)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()

# A helper function to read the latest model from disk.
@csrf_exempt
def read_latest_state():
    global rl_model
    global batch_count
    global next_batch_update_count
    global batch_update_frequency
    global checkpoint_dir

    try:
        search_path = os.path.join(checkpoint_dir, '*.json')
        print('searching {0}'.format(search_path))
        file_list = glob.glob(search_path)

        print('Checkpoint dir: {0}'.format(checkpoint_dir))
        print('file_list: {0}'.format(file_list))
        
        if (len(file_list) > 0):
            latest_file = max(file_list, key=os.path.getctime)

            print('Attempting to read latest state from {0}'.format(latest_file))
            file_text = ''
            with open(latest_file, 'r') as f:
                file_text = f.read().replace('\n', '')
            checkpoint_json = json.loads(file_text)
            rl_model.from_packet(checkpoint_json['model'])
            batch_count = int(checkpoint_json['batch_count'])
            next_batch_update_count = batch_count + batch_update_frequency
            print('Read latest state from {0}'.format(latest_file))
    finally:
        sys.stdout.flush()
        sys.stderr.flush()

# A helper function to parse environment variables
@csrf_exempt
def parse_parameters():
    global checkpoint_dir
    global batch_update_frequency

    try:
        checkpoint_dir = os.path.join(os.path.join(os.environ['data_dir'], 'checkpoint'), os.environ['experiment_name'])
        
        print('Checkpoint dir is {0}'.format(checkpoint_dir))
        
        if not os.path.isdir(checkpoint_dir):
            try:
                os.makedirs(checkpoint_dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        
        print('checkpoint_dir is {0}'.format(checkpoint_dir))
        batch_update_frequency = int(os.environ['batch_update_frequency'])
        print('batch_update_frequency is {0}'.format(batch_update_frequency))
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        
# On startup, the trainer node should identify itself to the agents by writing it IP address to (data_dir)\trainer_ip\(experiment_name)\trainer_ip.txt
@csrf_exempt
def write_ip():
    try:
        file_dir = os.path.join(os.path.join(os.environ['data_dir'], 'trainer_ip'), os.environ['experiment_name'])
        
        print('Writing to {0}...'.format(file_dir))
        
        if not os.path.isdir(file_dir):
            try:
                os.makedirs(file_dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        
        with open(os.path.join(file_dir, 'trainer_ip.txt'), 'w') as f:
            print('writing ip of {0}'.format(os.environ['AZ_BATCH_NODE_LIST'].split(';')[0]))
            f.write(os.environ['AZ_BATCH_NODE_LIST'].split(';')[0])
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
    
# stdout / stderr have already been redirected in manage.py
print('-----------STARTING TRAINER---------------')
print('-----------STARTING TRAINER---------------e', file=sys.stderr)

# Identify this node as a trainer, and kill all running instances of AirSim
os.system('DEL D:\\*.agent')
os.system('START "" powershell.exe D:\\AD_Cookbook_AirSim\\Scripts\\DistributedRL\\restart_airsim_if_agent.ps1')
sys.stdout.flush()
sys.stderr.flush()

# Initialize the node and notify agent nodes.
parse_parameters()
read_latest_state()
write_ip()
