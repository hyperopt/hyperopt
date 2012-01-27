
Vision for Hyperopt
===================


Hyperopt is based on a SON representation of functions. SON
representations have several features and benefits:

 * can be stored and manipulated in databases

 * are very flexible

 * genson is easy to type, can specify distributions over them

 * hyperopt algorithms are suited to search SON function spaces


Hyperopt should include genson directly or make it a dependency.

Hyperopt should include a function that maps SON structures to
scalar-valued functions.

The "Bandits" should be made plugins / layers rather than classes.
 
