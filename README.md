# Python-Networkx-Code
The code to run network analysis on the Shakeosphere data, esp: degree, betweenness, GCC
This directory contains files with data describing networks of printers,
publishers, authors, and booksellers beginning in the 15th century. The
Python program, graph_analytics.py, generates reports containing various
statistics from this data.

To run the program, make sure you are inside the directory containing
graph_analytics.py and db_shakeosphere.py and type the command:
python graph_analytics.py [time]
Where you replace [time] with either a single year (e.g., 1600) or a
dash-delimited time interval (e.g., 1600-1610).  The program will then
print out reports on various statistics, including Degree Rank, Weighted Degree Rank
Normalized Betweenness Centrality Rank, Normalized Edge Betweenness Centrality Rank,
Degree Distribution of the Largest Connected Component.

There are a few other optional inputs you can specify (basically ways to
turn off different reports).  For example, you can turn off the ranking
report with the flag --rank-off, e.g.,
python graph_analytics.py 1600-1610 --rank-off
To see the complete list of options, run the command:
python graph_analytics.py --help

If you want to save the output of the report into a text file (and not
have to copy and paste whatever gets printed out), you can use >, the
output redirection operator.  For example:
python graph_analytics.py 1600-1610 --rank-off > report.txt
Then, you can just open up report.txt and all of the information will be
there.

Finally, graph_analytics.py is just a Python script, so you can open it
and inspect/modify the code for yourself. Some basic modifications of the 
code to run other metrics (Eigenvector Centrality, Link Prediction) can
be found elsewhere in the repository. 
