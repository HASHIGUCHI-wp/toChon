// Gmsh project created on Sun Sep 10 10:00:48 2023
SetFactory("OpenCASCADE");


num_e = DefineNumber[2, Name "Parameters/num_e"];
element_size = DefineNumber[1.5, Name "Parameters/element_size"];
T = DefineNumber[num_e * element_size, Name "Parameters/T"];
dx_mesh_f = DefineNumber[T / 16, Name "Parameters/dx_mesh_f"];

alpha_L = DefineNumber[6, Name "Parameters/alpha_L"];
alpha_H = DefineNumber[21, Name "Parameters/alpha_H"];
alpha_W = DefineNumber[4, Name "Parameters/alpha_W"];
alpha_l1 = DefineNumber[1.5, Name "Parameters/alpha_l1"];
alpha_l2 = DefineNumber[2.5, Name "Parameters/alpha_l2"];
alpha_h1 = DefineNumber[4, Name "Parameters/alpha_h1"];
alpha_h2 = DefineNumber[15, Name "Parameters/alpha_h2"];
alpha_w = DefineNumber[2, Name "Parameters/alpha_w"];

L = DefineNumber[alpha_L * T, Name "Parameters/L"];
H = DefineNumber[alpha_H * T, Name "Parameters/H"];
W = DefineNumber[alpha_W * T, Name "Parameters/W"];
l1 = DefineNumber[alpha_l1 * T, Name "Parameters/l1"];
l2 = DefineNumber[alpha_l2 * T, Name "Parameters/l2"];
h1 = DefineNumber[alpha_h1 * T, Name "Parameters/h1"];
h2 = DefineNumber[alpha_h2 * T, Name "Parameters/h2"];
w = DefineNumber[alpha_w * T, Name "Parameters/w"];

Box(1) = {T, T, T, l1 + l2, w, h1};
Box(2) = {T, T, T + h1, l1, w, h2};

Transfinite Curve {9:12} = Ceil((l1 + l2) / dx_mesh_f) + 1 Using Progression 1;
Transfinite Curve {21:24} = Ceil((l1) / dx_mesh_f) + 1 Using Progression 1;
Transfinite Curve {1,3,5,7} = Ceil((h1) / dx_mesh_f) + 1 Using Progression 1;
Transfinite Curve {13,15,17,19} = Ceil((h2) / dx_mesh_f) + 1 Using Progression 1;
Transfinite Curve {2,4,6,8} = Ceil((w) / dx_mesh_f) + 1 Using Progression 1;
Transfinite Curve {14,16,18,20} = Ceil((w) / dx_mesh_f) + 1 Using Progression 1;
// Transfinite Curve {10, 12, 11, 9} = (l1 + l2) / dx_mesh_f + 1 Using Progression 1;

Transfinite Surface "*";
Transfinite Volume "*";

Printf("dx_mesh_f %f", dx_mesh_f);