# The Autonomous Driving Cookbook (Preview)



------

#### **NOTE:**

This project is developed and being maintained by the Microsoft Deep Learning and Robotics Garage Chapter. This is currently a work in progress. We will continue to add more tutorials and scenarios based on requests from our users and the availability of our collaborators.

------



## Simulation and AI

Autonomous Driving has transcended far beyond being a crazy moonshot idea over the last half decade or so. It is quickly becoming the biggest technology today that promises to shape our tomorrow, not very unlike when cars came into existence in the first place. Almost every single car manufacturer, every big technology company, and a number of very promising startups have been working on different aspects of autonomous driving to help shape this revolution. Some of the biggest drivers powering this change have been the recent advances in software (robotics and deep learning techniques), hardware technology (GPUs, FPGAs etc.) and cloud computing. Cloud platforms like [Azure](https://azure.microsoft.com) have enabled ingest and processing of large amounts of data, making it possible for companies to push for levels 4 and 5 of AD autonomy. 

Achieving those levels of autonomy however, is no easy feat. Despite the large amount of data collected every day, it is still insufficient to meet the demands of the ever increasing AI model complexity required by autonomous vehicles. The only way to collect such huge amounts of data is through the use of simulation. Simulation makes it easy to not only collect data from a variety of different scenarios which would take days, if not months in the real world (like different weather conditions, varying daylight etc.), it also provides a safe test bed for trained models. With Behavioral Cloning, you can easily prepare highly efficient models in simulation and fine tune them using a relatively low amount of real world data. Then there are models built using techniques like Reinforcement Learning, which can only be trained in simulation. With simulators such as [AirSim](https://github.com/Microsoft/AirSim), working on these scenarios has become very easy.

We believe that the best way to make a technology grow is by making it easily available and accessible to everyone. This is best achieved by making the barrier of entry to it as low as possible. At Microsoft, our mission is to empower every person and organization on the planet to achieve more. That has been our primary motivation behind preparing this cookbook. Our aim with this project is to help you get quickly acquainted and familiarized with different onboarding scenarios in autonomous driving so you can take what you learn here and employ it in your everyday job with a minimal barrier to entry.

### Who is this cookbook for?

Our plan is to make this cookbook a valuable resource for beginners, researchers and industry experts alike. We would love to hear your feedback on how we can evolve this project to reach that goal. Please use the GitHub Issues section to get in touch with us regarding ideas and suggestions.

### Tutorials available

Currently, the following tutorials are available:

- [Autonomous Driving using End-to-End Deep Learning: an AirSim tutorial](./AirSimE2EDeepLearning/)

Following tutorials will be available soon:

- Distributed Reinforcement Learning using AirSim and Batch AI
- Collecting data from your car for autonomous driving

### Contributing

Please read the [instructions and guidelines for collaborators](https://github.com/Microsoft/AutonomousDrivingCookbook/blob/master/CONTRIBUTING.md) if you wish to add a new tutorial to the cookbook. 

This project welcomes and encourages contributions and suggestions.  Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
