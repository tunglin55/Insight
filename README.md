# Insight Data Science - Traffic Buster
Being stuck in traffic can be extremly frustrating. The increased carbon dioxide emissions associated with congestions have also been shown to pose health threats to the general population. Moreover, the constant stop-and-go motions also attract greater chances of collisions. On top of all this, traffic delays are costly not only to our individual households, but also to cities on the scale of hundreds of the millions. Thus, to be able to predict traffic presents multiple significant advantages. For example, cities can better allocate resources for designing smoother routes; delivery and rideshare companies can be more on-time with their services; advertisement companies can predict hubs of traffic to place their ads. Thus, the purpose of this project is to generate a model that can predict traffic in downtown Toronto.

## Travel Time Data 
Travel time data on streets and highways across the city of Toronto were used for this project. Specifically, data were recorded from Bluetooth and Wifi sensors at 5-minute intervals. Information were derived from single locations whenever a mobile deivce or vehicle passes by the sensors.

## Features
Several features were narrowed down to predict travel times: Day of the week, Hour of the day, Holidays, Weather conditions (temperature, visibility, and snow presence), Toronto Raptors', Blue Jays' and Maple Leafs' games and events hosted at the Scotiabank arena. 

## Models
Random forrest regressions were used to predict travel times in downtown Toronto. Datasets were split 75% and 25% into train and test datasets respectively. Hyperparameter tuning was performed where a random search was first initiated. Subsequently, 

## Webapp
A web app was built with Flask and hosted on AWS that allows users to predict delays based on their date and time of travel in downtown Toronto. A map with travel times in different segments of the city is displayed. Click [here](http://www.torontotrafficforecast.com) to access the Webapp.

## Generalizations 

