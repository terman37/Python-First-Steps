# First VS Python script (20191206)



- FirstPython-A19 : basic python stuff
- CleanPython-A19: 
  - Lambda programming
  - generators
  - decorators
  - 
- UseCase_TSP_naive_solver (too long with nb points > 10):

##### USE CASE:

> Deal with a Polygonal Chain:
> 	 Each point has (x,y)
> 	 2 points can make:
>
> 	  - a straight line
> 	  - a line segment: defined length (norm)
>
> **Traveling Salesman Problem:**
> 	what is the shortest path to connect all the points
> 	here bounded TSP: points randomly distributed within a bound Euclidian rectangle.
> 	Naïve solver = O(n!) complexity