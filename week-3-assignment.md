Week 3 Assignment
========================================================
author: Sami Darko
date: "18/04/2019"
autosize: true


Plot Code
========================================================

<br />
<br />

```r
library(plotly)

p <- plot_ly(
  x = rnorm( 1000 ), 
  y = rnorm( 1000 ), 
  mode = 'markers' )

htmlwidgets::saveWidget(
  as.widget(p), 
  file = "week-3-plot.html")
```
<br />
<br />
Plotly documentation [here](https://plot.ly/r/reference/)


A plot with Plotly
========================================================

<iframe src="week-3-plot.html" style="position:absolute;height:100%;width:100%"></iframe>
