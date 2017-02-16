WhatHappenedThere?
==================

[WhatHappenedThere?](http://whathappenedthere.xyz/) is a tool that matches Wikipedia traffic spikes with breaking news. A spike in traffic signals an increased interest in the webpage. This app helps pointing out the reasons beahind it by connecting spikes and events. After you input your topic of interest, the app finds the relative Wikipedia page, downloads the time series and automatically detects the spikes. It then queries a database storing New York Times articles and returns the ones deemed the most relevant by a natural language processing algorithm. For more details, please check out the [blog post](http://gobboph.github.io/blog/wht/) I wrote about it

This repo contains all the project, but the data. The data could be easily downloaded using the New York Times API and the the functions defined in the *nyt* module. In order to do that you will need your own NYT API key. [Here](https://developer.nytimes.com/) are the details on how to get one and the documentation.

Play with the app and let me know what you think!