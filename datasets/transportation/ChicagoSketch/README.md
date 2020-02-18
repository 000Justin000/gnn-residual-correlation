# Network
A fairly realistic yet aggregated representation of the Chicago region.  The original data is known to provide low levels of congestion, not realistic for the Chicago region. For algorithm testing it is recommended to double the original trip table.

## Source  
Developed and provided by the Chicago Area Transportation Study. 
Via: http://www.bgu.ac.il/~bargera/tntp/


## Scenario


## Contents

 - `ChicagoSketch_net.tntp` Network  
 - `ChicagoSketch_trips.tntp` Demand  
 - `ChicagoSketch_node.tntp` Node coordinates in Illinois state plane coordinate system, feet
 - `ChicagoSketch_flow.tntp`  Best known flow solution   

## Dimensions  
Zones: 387
Nodes: 933
Links: 2950
Trips: 1,260,907.4400005303

## Units
Time: minutes
Distance: miles
Speed: 
Cost: cents 
Coordinates: Illinois State Plane, feet

## Generalized Cost Weights
Toll: 0.02 minutes/cent
Distance: 0.04 minutes/mile

## Solutions

`ChicagoSketch_flow.tntp` is the best known flow solution with Average Excess Cost of 2.1E-13. Optimal objective function value: 17313018.7387477. 

## Known Issues
FIXME translate to Github Issues

## References

The preparation of this dataset is described in the following references:  
 - Eash, R.W., K.S. Chon, Y.J. Lee and D.E. Boyce (1983) Equilibrium Traffic Assignment on an Aggregated Highway Network for Sketch Planning, Transportation Research Record, 994, 30-37.  
 - Boyce, D.E., K.S. Chon, M.E. Ferris, Y.J. Lee, K-T. Lin and R.W. Eash (1985) Implementation and Evaluation of Combined Models of Urban Travel and Location on a Sketch Planning Network, Chicago Area Transportation Study, xii + 169 pp.  
