model Pendelmodel

  parameter Real m=1; // mass of pendulum
  parameter Real l=1; // length of pendulum
  parameter Real g=9.81; // gravitational constant
  
  Real u; // input torque
  Real th(start=1); // angular displacement
  Real thdot(start=0); // angular velocity
  
equation
  
  der(th) = thdot;
  der(thdot) = (u-m*g*l*sin(th))/(m*l^2);
  u=0;


end Pendelmodel;
