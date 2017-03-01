WhatHappenedThere?
==================

[WhatHappenedThere?](http://whathappenedthere.xyz/) is a tool that matches Wikipedia traffic spikes with breaking news. A spike in traffic signals an increased interest in the webpage. This app helps to point out the reasons behind the increased interest by connecting spikes and events. When you input a topic of interest, the app finds the relevant Wikipedia page, downloads the time series and automatically detects the spikes in traffic. It then queries a database of New York Times articles and returns those articles deemed the most relevant by a natural language processing algorithm. For more details plese check out the [blog post](http://gobboph.github.io/blog/wht/) about this project.

This repo contains all of the code for the project, but does not include the data. The data could be easily downloaded using the New York Times API and the the functions defined in the nyt.py file. In order to do that you will need your own NYT API key. [Here](https://developer.nytimes.com/) are the details on how to get one and the documentation for the API.

In the application folder, you find the run.py file and the flasknews folder, which contains the backbone of the app.

The views.py file in the flasknews folder contains the python flask code, whereas anomalies.py, nyt.py and wiki.py contain the necessary functions. More precisely, wiki.py contains the function to query Wikipedia for the time series, anomalies.py contains the spike detection algorithm and nyt.py contains the functions to query for the articles and perform natural language processing. nyt.py also contains some functions to download articles from the New York Times and The Guardian.

Notice that to make the app work as is, it is necessary to set up a SQL database with all the articles from the New York Times starting July 2015 (the earliest date in the Wikipedia time series). This database is where the function in nyt.py looks for the articles when needed.

The templates folder quite simply contains the templates of the html pages and the static folder contains the css, js and fonts related files.

Play with the app and let me know what you think!