# Instructions and guidelines for collaborators 

We truly appreciate your contribution to help expand and improve the Autonomous Driving Cookbook. At this time, we are accepting contributions in the form of full-length tutorials only. If you wish to propose changes to an existing tutorial, please use the GitHub Issues section. 

The main motivation behind this cookbook is to create a collection of tutorials that can benefit beginners, researchers and industry experts alike. Please ensure your tutorial is prepared in the same vein with similar objectives. Your tutorial can cater to either or all of these audiences. The purpose of tutorials in this cookbook is not to demonstrate a cutting edge research technique or to promote a product, but instead to reduce barriers to entry for people who are getting started with, or are already working in the field of Autonomous Driving. Your tutorial should most definitely leverage new research techniques and/or products, but they should not be the main focus. The emphasis needs to be on new methods and techniques readers can learn by working on your tutorial, and how they can use and expand on them to help them achieve their individual goals.

If you wish to add a new tutorial to the cookbook, please follow the following steps.

## Step 1

1. Make sure you have read the [Contributing](./README.md#contributing) section in the main README. 
2. Create a new GitHub Issue, using the 'new tutorial proposal' label. Please provide the following information:
   1. Title of the tutorial
   2. A brief description (2-3 sentences)
   3. An email address for our team to reach out to you

## Step 2

1. Someone from our team will get in touch with you over email to request a one-page write-up for your proposed tutorial. Please make sure your one pager includes the following information:
   1. Title of the proposed tutorial
   2. List of authors with affiliations
   3. Abstract
   4. Proposed format of the tutorial (e.g. Python notebooks, single readme with code snippets etc.)
   5. Justification for adding the tutorial to the cookbook: does it cover a topic currently not included in the cookbook?
   6. List of technologies the tutorial uses
   7. Target audience for the tutorial
2. Once we receive your one-pager, our team will work with you to get any additional details and provide suggestions as needed. 

## Step 3

If the team decides to move forward with adding the proposed tutorial to the cookbook, we will work with you to prepare the tutorial on a new branch which will be merged to main once the tutorial is ready.

While working on your tutorial, please make sure of the following:

1. Your entire tutorial, and any related files should sit inside a single folder in the main repo.
2. Any non-relevant local files should not be checked in. Use a .gitignore file to help with this.
3. Any data needed for the tutorial should not be checked in. There should instead be download links provided to your dataset(s) from within the tutorial, wherever necessary. If you are using a dataset not owned by you, please make sure you have the necessary permissions and that you acknowledge the owners appropriately.
4. Your tutorial needs to have a README.md file with the following sections, as necessary:
   1. **Title of the tutorial**
   2. **Authors and affiliations**
   3. **Overview:** This section should establish the purpose of the tutorial in 3-5 sentences. It should also tell the readers what they can expect to achieve after finishing the tutorial.
   4. **Structure of the tutorial:** Use this section to describe how the tutorial is set up and laid out and where the reader should get started.
   5. **Prerequisites and setup:**
      1. Background needed
      2. Environment setup, if any
      3. Hardware setup, if any
      4. Information about datasets used, if any
      5. Additional notes
   6. **References**, if any
5. Please make sure to appropriately acknowledge any references you use in the tutorial. You can use the **References** section in the README for this, or you can simply link to the referred material directly from the tutorial content.



