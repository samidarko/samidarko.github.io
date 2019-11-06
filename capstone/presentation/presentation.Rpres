Coursera Data Science Capstone Project
========================================================
author: Vincent Dupont
date: 4th November 2019
autosize: true

Introduction
========================================================

This capstone is the last course of a serie of 10. 

Students are required to use the skills they've been learning during the cursus to solve a problem they never met before: word prediction based on the user input.

For the purpose we are going to use technics as such as text analysis and Natural Language Processing.

As the teachers decided to give us a project on a subject we didn't study I decided I could use a different technology to solve the problem.

Data and model
========================================================

We are give a data set with a lot texts extracted from internet such as posts, comments or twitts. 

A quite exhaustive explanation on how the data was processed and the model to use is given the project milestone:
https://samidarko.github.io/capstone/milestone_report

The model only support 1, 2 and 3-gram but it is extensible.

The application
========================================================

This application is not a shiny app but [React](https://reactjs.org/) front-end with a [Flask](https://www.palletsprojects.com/p/flask/) api for the backend.

The 2 applications are hosted on an [AWS](https://aws.amazon.com/) EC2 instance. The front-end is delivered by an [nginx](https://www.nginx.com) web server and the backend is running with [uwsgi](https://uwsgi-docs.readthedocs.io/en/latest/) + [supervisord](http://supervisord.org/).

Using the application is pretty simple:
 - Start to type and it will predict a word starting with characters you typed
 - Add a space to word you typed and it will try to guess the next word

My tradeoff is to use memory instead of computing. I have a couple of "fat models" (monogram, bigram and trigram). The application is using around 250Mb.


Conclusion
========================================================

All the code can be found here:
https://github.com/samidarko/datascience-capstone/tree/master/app

The application is running there:
http://ec2-52-77-230-203.ap-southeast-1.compute.amazonaws.com/
(sorry for poor design)



