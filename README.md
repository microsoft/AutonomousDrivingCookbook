# The Autonomous Driving Cookbook (Preview)



------

#### **NOTE:**

This project is developed and being maintained by the Microsoft Deep Learning and Robotics Garage Chapter. This is currently a work in progress. We will continue to add more tutorials and scenarios based on requests from our users and the availability of our collaborators.

------
<p align="center">
  <img src="AirSimE2EDeepLearning/car_driving.gif?raw=true" />
</p>


Autonomous driving has transcended far beyond being a crazy moonshot idea over the last half decade or so. It has quickly become one of the biggest technologies today that promises to shape our tomorrow, not very unlike when cars first came into existence. A big driver powering this change is the recent advances in software (Artificial Intelligence), hardware (GPUs, FPGAs etc.) and cloud computing, which have enabled ingest and processing of large amounts of data, making it possible for companies to push for levels 4 and 5 of autonomy. Achieving those levels of autonomy though, require training on hundreds of millions and sometimes hundreds of billions of miles worth of training data to demonstrate reliability, according to a [report](https://www.rand.org/pubs/research_reports/RR1478.html) from RAND.

Despite the large amount of data collected every day, it is still insufficient to meet the demands of the ever increasing AI model complexity required by autonomous vehicles. One way to collect such huge amounts of data is through the use of simulation. Simulation makes it easy to not only collect data from a variety of different scenarios which would take days, if not months in the real world (like different weather conditions, varying daylight etc.), it also provides a safe test bed for trained models. With behavioral cloning, you can easily prepare highly efficient models in simulation and fine tune them using a relatively low amount of real world data. Then there are models built using techniques like Reinforcement Learning, which can only be trained in simulation. With simulators such as [AirSim](https://github.com/Microsoft/AirSim), working on these scenarios has become very easy.

<p align="center">
  <img src="DistributedRL/car_driving_2.gif?raw=true" />
</p>


We believe that the best way to make a technology grow is by making it easily available and accessible to everyone. This is best achieved by making the barrier of entry to it as low as possible. At Microsoft, our mission is to empower every person and organization on the planet to achieve more. That has been our primary motivation behind preparing this cookbook. Our aim with this project is to help you get quickly acquainted and familiarized with different onboarding scenarios in autonomous driving so you can take what you learn here and employ it in your everyday job with a minimal barrier to entry.

### Who is this cookbook for?

Our plan is to make this cookbook a valuable resource for beginners, researchers and industry experts alike. Tutorials in the cookbook are presented as [Jupyter notebooks](http://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html), making it very easy for you to download the instructions and get started without a lot of setup time. To help this further, wherever needed, tutorials come with their own datasets, helper scripts and binaries. While the tutorials leverage popular open-source tools (like Keras, TensorFlow etc.) as well as Microsoft open-source and commercial technology (like AirSim, Azure virtual machines, Batch AI, CNTK etc.), the primary focus is on the content and learning, enabling you to take what you learn here and apply it to your work using tools of your choice.  

We would love to hear your feedback on how we can evolve this project to reach that goal. Please use the GitHub Issues section to get in touch with us regarding ideas and suggestions.

### Tutorials available

Currently, the following tutorials are available:

- [Autonomous Driving using End-to-End Deep Learning: an AirSim tutorial](./AirSimE2EDeepLearning/)
- [Distributed Deep Reinforcement Learning for Autonomous Driving](./DistributedRL/)

Following tutorials will be available soon:

- Lane Detection using Deep Learning

### Contributing

Please read the [instructions and guidelines for collaborators](https://github.com/Microsoft/AutonomousDrivingCookbook/blob/master/CONTRIBUTING.md) if you wish to add a new tutorial to the cookbook. 

This project welcomes and encourages contributions and suggestions.  Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
