# Extended-Modified-RelexNet

This project is based on [LSX-Sneakerprogrammer](https://github.com/LSX-Sneakerprogrammer)/[Modified-RelexNet](https://github.com/LSX-Sneakerprogrammer/Modified-RelexNet) project.

## Files
- [relexnet_classifier_with_explaination](RelexNet%20with%20Modifiers/relexnet_classifier_with_explaination.py), [data_loader](RelexNet%20with%20Modifiers/data_loader.py) and [training](RelexNet%20with%20Modifiers/training.py) are modified from the base repo in order to output data graph.
- [graphHandler](RelexNet%20with%20Modifiers/graphHandler.py) is used for creating and managing data graph.
- [explanator](RelexNet%20with%20Modifiers/explanator.py) is used for loading and querying the data graph.
- [app](RelexNet%20with%20Modifiers/app.py) is the simple app for explanation visualisation.

## Start-up

### Data graph generation

- If you want to train and export the model you can run `python relexnet_classifier_with_explaination.py --config configs/config-train.json`. The config file have most of the parameters for training the model.
- After the model trained you can run [relexnet_classifier_with_explaination](RelexNet%20with%20Modifiers/relexnet_classifier_with_explaination.py) with [config.json](RelexNet%20with%20Modifiers/configs/config.json). The data graph will save to datagraphs directory.

### Visualisation

Run `flask run` in RelexNet with Modifiers directory will start up the develop web server for the explanation visualization. You can access the page on [http://localhost:5000](http://localhost:5000). You need to have [flask](https://flask.palletsprojects.com/en/2.1.x/installation/) package installed to run the app.

---

Below is the README of the base project from [LSX-Sneakerprogrammer](https://github.com/LSX-Sneakerprogrammer)/[Modified-RelexNet](https://github.com/LSX-Sneakerprogrammer/Modified-RelexNet)

---

# Modified-RelexNet

This is the project based on RelexNet. There are three models under the file, the modified RelexNet, the modified RelexNet with Negative words and Modifiers, the modified RelexNet without Modifiers.

# Start-up

If you have enough memory (perhaps more than 25gb), you could download the three files and run the file relexnet_classifier.py (the three files all have this .py file)

For me, because of the limit memory size, I used Google Colab pro (could provide 25gb memory) to run the three models. If you use Colab pro, you could run the .ipynb file under each three file. In order to Loading data, you need to first upload the whole project to Google Drive. Then, run .ipynb file, which helps get access to dataset and run the relexnet_classifier.py file.

