(* ::Package:: *)

(* ::Title:: *)
(*O(N1) x O(N2) model at finite temperature, LPA: the potential of two invariants*)


(* ::Text:: *)
(*Flow of the effective potential U(rho1, rho2), rho_a = phi_a^2/2 for an O(N1) and an O(N2) multiplet, in d=3+1 with 3d (spatial) regulators. The Matsubara sum is done numerically by Integrator_fT_p2, so the summand stays rational in the curvatures and no eigenvalues are needed.*)
(*Run headless with:  wolfram -script ONxOM.wl*)


(* ::Input::Initialization:: *)
Get["DiFfRG`"]
SetDirectory[If[$InputFileName==="",Directory[],DirectoryName[$InputFileName]]];


(* ::Chapter:: *)
(*Flow*)


(* ::Text:: *)
(*With q = q0^2 + q^2 + R_B the meson propagators are the N1-1 Goldstones of the first multiplet (mass U_1), the N2-1 of the second (U_2), and the radial 2x2 block*)
(*  M = {{U_1 + 2 rho1 U_11, 2 sqrt(rho1 rho2) U_12}, {2 sqrt(rho1 rho2) U_12, U_2 + 2 rho2 U_22}} = {{M11, M12}, {M12, M22}}.*)
(*The Wetterich equation, d_t U = -F, gives F = 1/2 T sum_n int d^3q/(2pi)^3 dot(R_B) Tr G with Tr (q + M)^-1 = (2q + M11 + M22)/((q + M11)(q + M22) - M12^2).*)
(*V_ON is the flow of one invariant rho = rho1 + rho2, the O(N1 + N2) model, for comparison in the symmetric limit.*)


(* ::Input::Initialization:: *)
q=l0^2+l1^2+RB[k^2,l1^2];
flowV=1/2 RBdot[k^2,l1^2]((N1-1)/(q+U1)+(N2-1)/(q+U2)+(2q+M11+M22)/((q+M11)(q+M22)-M12^2));
flowVON=1/2 RBdot[k^2,l1^2]((N-1)/(q+m2Pi)+1/(q+m2Sigma));


(* ::Chapter:: *)
(*Code generation*)


(* ::Text:: *)
(*U_a and M_ab depend on the solution, so they are AD parameters. Both kernels are evaluated for all points of a KT batch in one map_points call (and per point through get(), for the per-point path of the assembler).*)


(* ::Input::Initialization:: *)
MakeFT[expr_,name_,parameters_]:=MakeKernel[expr,
	"Name"->name,
	"Integrator"->"Integrator_fT_p2",
	"d"->4,
	"AD"->True,
	"Device"->"TBB",
	"ComputeType"->"double",
	"MapPoints"->True,
	"Parameters"->parameters,
	"IntegrationVariables"->{"l1","l0"}
];
MakeFT[flowV,"V",{<|"Name"->"k"|>,<|"Name"->"N1"|>,<|"Name"->"N2"|>,<|"Name"->"U1","AD"->True|>,<|"Name"->"U2","AD"->True|>,
	<|"Name"->"M11","AD"->True|>,<|"Name"->"M12","AD"->True|>,<|"Name"->"M22","AD"->True|>}];
MakeFT[flowVON,"V_ON",{<|"Name"->"k"|>,<|"Name"->"N"|>,<|"Name"->"m2Pi","AD"->True|>,<|"Name"->"m2Sigma","AD"->True|>}];
UpdateFlows["ONxOMFlows"]
