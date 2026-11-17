#include <cmath>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include "fluid.h"

#include <map>
#include <string>
#include <functional>

#include <omp.h>
#include <algorithm>

#include <chrono>
#include <cassert>

void launch_advect_kernel(int blocks, int threads, std::vector<FluidCell*> leaves, int step, bool& minSizeSmall, long double& thisMaxDensity, long double& maxDensity, long double& totMass) {
	if (step == 1) {
		advect_kernel<<<blocks, threads>>>(leaves, minSizeSmall, thisMaxDensity, maxDensity);
	} else if (step == 2) {
		advect_kernel2<<<blocks, threads>>>(leaves);
	} else if (step == 3) {
		advect_kernel3<<<blocks, threads>>>(leaves, minSizeSmall, thisMaxDensity, maxDensity);
	} else if (step == 4) {
		advect_kernel4<<<blocks, threads>>>(leaves, totMass);
	}
	cudaDeviceSynchronize();
}

__global__ void advect_kernel(std::vector<FluidCell*> leaves, bool& minSizeSmall, long double& thisMaxDensity, long double& maxDensity) {
	int workIdx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (workIdx < leaves.size()) {
		FluidCell *cell = leaves[workIdx];
		FluidCell *cellR = cell->getRight();
		FluidCell *cellL = cell->getLeft();
		FluidCell *cellT = cell->getTop();
		FluidCell *cellB = cell->getBottom();
		VelocityVector vC = cell->getVelocity();
		float X = cell->getX();
		float Y = cell->getY();
		float Z = cell->getZ();

		if (cell->getDensity() > 1e-2*maxDensity && std::max(abs(vC.getVx()),abs(vC.getVy())) > maxV) {
			maxV = std::max(abs(vC.getVx()),abs(vC.getVy()));
		}
		float fracTraveled = std::max(abs(vC.getVx()),abs(vC.getVy()))*getdt()/std::sqrt(cell->getSize());
		if (fracTraveled > maxFracTraveled) {
			if (fracTraveled > 1) {
				// std::cout << "maxFrac: " << fracTraveled << " " << (cell->getMass()-cellR->getMass())/(2*cell->getWidth()) << std::endl;
			}
			maxFracTraveled = fracTraveled;
		}

		// avgFracTraveled += std::isfinite(fracTraveled) ? fracTraveled : 0;

		if (cell->getHeight() == minSize || cell->getWidth() == minSize) minSizeSmall = false;

		long double massFluxR = 0;
		long double massFluxL = 0;
		long double massFluxT = 0;
		long double massFluxB = 0;
		long double EFluxR = 0;
		long double EFluxL = 0;
		long double EFluxT = 0;
		long double EFluxB = 0;
		long double vxFluxR = 0;
		long double vyFluxR = 0;
		long double vxFluxL = 0;
		long double vyFluxL = 0;
		long double vxFluxT = 0;
		long double vyFluxT = 0;
		long double vxFluxB = 0;
		long double vyFluxB = 0;
		long double XFluxR = 0;
		long double XFluxL = 0;
		long double XFluxT = 0;
		long double XFluxB = 0;
		long double YFluxR = 0;
		long double YFluxL = 0;
		long double YFluxT = 0;
		long double YFluxB = 0;
		long double ZFluxR = 0;
		long double ZFluxL = 0;
		long double ZFluxT = 0;
		long double ZFluxB = 0;
		long double unFluxR = 0;
		long double unFluxL = 0;
		long double unFluxT = 0;
		long double unFluxB = 0;
		long double vxR;
		long double vxL;
		long double vyT;
		long double vyB;
		VelocityVector vR, vL, vT, vB;
		long double massR, massL, massT, massB;
		long double ER, EL, ET, EB;
		long double mass = cell->getMass();
		long double E = cell->getE(false);
		float XR, XL, XT, XB, YR, YL, YT, YB, ZR, ZL, ZT, ZB;

		if (cell->getDensity() > thisMaxDensity) thisMaxDensity = cell->getDensity();
		vxR = (cell->getRight()->getVelocity().getVx() + vC.getVx())/2;
		vxL = (cell->getLeft()->getVelocity().getVx() + vC.getVx())/2;
		vyT = (cell->getTop()->getVelocity().getVy() + vC.getVy())/2;
		vyB = (cell->getBottom()->getVelocity().getVy() + vC.getVy())/2;
		XR = cellR->getX();
		XL = cellL->getX();
		XT = cellT->getX();
		XB = cellB->getX();
		YR = cellR->getY();
		YL = cellL->getY();
		YT = cellT->getY();
		YB = cellB->getY();
		ZR = cellR->getZ();
		ZL = cellL->getZ();
		ZT = cellT->getZ();
		ZB = cellB->getZ();
		// vR = cellR->getVelocity();
		// vL = cellL->getVelocity();
		// vT = cellT->getVelocity();
		// vB = cellB->getVelocity();
		Direction dir;
		FluidCell *nextCell;

		long double dist, face;

		// not a boundary
		if (cellR != cell) {
			if (cellR->hasChildren()) {
				// vxR = 0;
				// massR = 0;
				FluidCell *children[2] = {cellR->nw, cellR->sw};
				for (int i = 0; i < 2; i++) {
					vxR = (children[i]->getVelocity().getVx() + vC.getVx())/2;
					massR = children[i]->getMass();
					ER = children[i]->getE(false);
					if (vxR < 0) {
						dir = RIGHT;
						nextCell = children[i];
						dist = std::abs(vxR*getdt());
					} else {
						dir = LEFT;
						nextCell = cell;
						dist = std::min(std::abs(vxR*getdt()),cell->getWidth());
					}
					face = children[i]->getHeight();
					
					massFluxR += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass();}, dir, 0, false, false);
					EFluxR += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getE(false)*c->getSize();}, dir, 0, false, false, [](FluidCell *c, long double m){c->newE -= m/c->getSize();});
					// vxFluxR += getFlux(nextCell, std::abs(vxR*getdt()), children[i]->getHeight(), MOMENTUM, dir, 0);
					vxFluxR += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass()*c->getVelocity().getVx();}, dir, 0);
					vyFluxR += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass()*c->getVelocity().getVy();}, dir, 0);
					XFluxR += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getX()*c->getMass();}, dir, 0);
					YFluxR += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getY()*c->getMass();}, dir, 0);
					ZFluxR += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getZ()*c->getMass();}, dir, 0);
					unFluxR += getFlux2(nextCell, dist, face, [](FluidCell *c) {return 1;}, dir, 0);
				}
				
			} else {
				// massR = cellR->getMass();
				// ER = cellR->getE(false);
				if (vxR < 0) {
					dir = RIGHT;
					nextCell = cellR;
					dist = std::abs(vxR*getdt());
				} else {
					dir = LEFT;
					nextCell = cell;
					dist = std::min(std::abs(vxR*getdt()),cell->getWidth());
				}
				face = cell->getHeight();

				massFluxR = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass();}, dir, 0, false, false);
				EFluxR = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getE(false)*c->getSize();}, dir, 0, false, false, [](FluidCell *c, long double m){c->newE -= m/c->getSize();});
				// vxFluxR = getFlux(nextCell, std::abs(vxR*getdt()), cell->getHeight(), MOMENTUM, dir, 0);
				vxFluxR = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass()*c->getVelocity().getVx();}, dir, 0);
				vyFluxR = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass()*c->getVelocity().getVy();}, dir, 0);
				XFluxR = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getX()*c->getMass();}, dir, 0);
				YFluxR = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getY()*c->getMass();}, dir, 0);
				ZFluxR = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getZ()*c->getMass();}, dir, 0);
				unFluxR = getFlux2(nextCell, dist, face, [](FluidCell *c) {return 1;}, dir, 0);
			}
			// std::cout << massFluxR << std::endl;
			if (vxR < 0) {
				massFluxR *= -1;
				EFluxR *= -1;
				vxFluxR *= -1;
				vyFluxR *= -1;
				XFluxR *= -1;
				YFluxR *= -1;
				ZFluxR *= -1;
				unFluxR *= -1;
			}
		}
		// not a boundary
		if (cellL != cell) {
			if (cellL->hasChildren()) {
				// vxL = 0;
				// massL = 0;
				FluidCell *children[2] = {cellL->ne, cellL->se};
				for (int i = 0; i < 2; i++) {
					vxL = (children[i]->getVelocity().getVx() + vC.getVx())/2;
					massL = children[i]->getMass();
					EL = children[i]->getE(false);
					if (vxL < 0) {
						dir = RIGHT;
						nextCell = cell;
						dist = std::min(std::abs(vxL*getdt()),cell->getWidth());
					} else {
						dir = LEFT;
						nextCell = children[i];
						dist = std::abs(vxL*getdt());
					}
					face = children[i]->getHeight();

					massFluxL += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass();}, dir, 0, false, false);
					EFluxL += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getE(false)*c->getSize();}, dir, 0, false, false, [](FluidCell *c, long double m){c->newE -= m/c->getSize();});
					// vxFluxL += getFlux(nextCell, std::abs(vxL*getdt()), children[i]->getHeight(), MOMENTUM, dir, 0);
					vxFluxL += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass()*c->getVelocity().getVx();}, dir, 0);
					vyFluxL += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass()*c->getVelocity().getVy();}, dir, 0);
					XFluxL += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getX()*c->getMass();}, dir, 0);
					YFluxL += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getY()*c->getMass();}, dir, 0);
					ZFluxL += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getZ()*c->getMass();}, dir, 0);
					unFluxL += getFlux2(nextCell, dist, face, [](FluidCell *c) {return 1;}, dir, 0);
				}
			} else {
				massL = cellL->getMass();
				EL = cellL->getE(false);
				if (vxL < 0) {
					dir = RIGHT;
					nextCell = cell;
					dist = std::min(std::abs(vxL*getdt()),cell->getWidth());
				} else {
					dir = LEFT;
					nextCell = cellL;
					dist = std::abs(vxL*getdt());
				}
				face = cell->getHeight();

				massFluxL = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass();}, dir, 0, false, false);
				EFluxL = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getE(false)*c->getSize();}, dir, 0, false, false, [](FluidCell *c, long double m){c->newE -= m/c->getSize();});
				// vxFluxL = getFlux(nextCell, std::abs(vxL*getdt()), cell->getHeight(), MOMENTUM, dir, 0);
				vxFluxL = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass()*c->getVelocity().getVx();}, dir, 0);
				vyFluxL = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass()*c->getVelocity().getVy();}, dir, 0);
				XFluxL = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getX()*c->getMass();}, dir, 0);
				YFluxL = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getY()*c->getMass();}, dir, 0);
				ZFluxL = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getZ()*c->getMass();}, dir, 0);
				unFluxL = getFlux2(nextCell, dist, face, [](FluidCell *c) {return 1;}, dir, 0);
			}
			if (vxL < 0) {
				massFluxL *= -1;
				EFluxL *= -1;
				vxFluxL *= -1;
				vyFluxL *= -1;
				XFluxL *= -1;
				YFluxL *= -1;
				ZFluxL *= -1;
				unFluxL *= -1;
			}
		}

		// not a boundary
		if (cellT != cell) {
			if (cellT->hasChildren()) {
				FluidCell *children[2] = {cellT->sw, cellT->se};
				for (int i = 0; i < 2; i++) {
					vyT = (children[i]->getVelocity().getVy() + vC.getVy())/2;
					massT = children[i]->getMass();
					ET = children[i]->getE(false);
					if (vyT < 0) {
						dir = UP;
						nextCell = children[i];
						dist = std::abs(vyT*getdt());
					} else {
						dir = DOWN;
						nextCell = cell;
						dist = std::min(std::abs(vyT*getdt()),cell->getHeight());
					}

					face = children[i]->getWidth();
					massFluxT += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass();}, dir, 0, false, false);
					EFluxT += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getE(false)*c->getSize();}, dir, 0, false, false, [](FluidCell *c, long double m){c->newE -= m/c->getSize();});
					vxFluxT += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass()*c->getVelocity().getVx();}, dir, 0);
					vyFluxT += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass()*c->getVelocity().getVy();}, dir, 0);
					XFluxT += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getX()*c->getMass();}, dir, 0);
					YFluxT += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getY()*c->getMass();}, dir, 0);
					ZFluxT += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getZ()*c->getMass();}, dir, 0);
					unFluxT += getFlux2(nextCell, dist, face, [](FluidCell *c) {return 1;}, dir, 0);
				}
				
			} else {
				massT = cellT->getMass();
				ET = cellT->getE(false);
				if (vyT < 0) {
					dir = UP;
					nextCell = cellT;
					dist = std::abs(vyT*getdt());
				} else {
					dir = DOWN;
					nextCell = cell;
					dist = std::min(std::abs(vyT*getdt()),cell->getHeight());
				}

				face = cell->getWidth();
				massFluxT = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass();}, dir, 0, false, false);
				EFluxT = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getE(false)*c->getSize();}, dir, 0, false, false, [](FluidCell *c, long double m){c->newE -= m/c->getSize();});
				vxFluxT = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass()*c->getVelocity().getVx();}, dir, 0);
				vyFluxT = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass()*c->getVelocity().getVy();}, dir, 0);
				XFluxT = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getX()*c->getMass();}, dir, 0);
				YFluxT = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getY()*c->getMass();}, dir, 0);
				ZFluxT = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getZ()*c->getMass();}, dir, 0);
				unFluxT = getFlux2(nextCell, dist, face, [](FluidCell *c) {return 1;}, dir, 0);
			}
			if (vyT < 0) {
				massFluxT *= -1;
				EFluxT *= -1;
				vxFluxT *= -1;
				vyFluxT *= -1;
				XFluxT *= -1;
				YFluxT *= -1;
				ZFluxT *= -1;
				unFluxT *= -1;
			}
		}
		// not a boundary
		if (cellB != cell) {
			if (cellB->hasChildren()) {
				// vyB = 0;
				// massB = 0;
				FluidCell *children[2] = {cellB->nw, cellB->ne};
				for (int i = 0; i < 2; i++) {
					vyB = (children[i]->getVelocity().getVy() + vC.getVy())/2;
					massB = children[i]->getMass();
					EB = children[i]->getE(false);
					if (vyB < 0) {
						dir = UP;
						nextCell = cell;
						dist = std::min(std::abs(vyB*getdt()),cell->getHeight());
					} else {
						dir = DOWN;
						nextCell = children[i];
						dist = std::abs(vyB*getdt());
					}

					face = children[i]->getWidth();
					massFluxB += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass();}, dir, 0, false, false);
					EFluxB += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getE(false)*c->getSize();}, dir, 0, false, false, [](FluidCell *c, long double m){c->newE -= m/c->getSize();});
					vxFluxB += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass()*c->getVelocity().getVx();}, dir, 0);
					vyFluxB += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass()*c->getVelocity().getVy();}, dir, 0);
					XFluxB += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getX()*c->getMass();}, dir, 0);
					YFluxB += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getY()*c->getMass();}, dir, 0);
					ZFluxB += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getZ()*c->getMass();}, dir, 0);
					unFluxB += getFlux2(nextCell, dist, face, [](FluidCell *c) {return 1;}, dir, 0);
				}
				
			} else {
				massB = cellB->getMass();
				EB = cellB->getE(false);
				if (vyB < 0) {
					dir = UP;
					nextCell = cell;
					dist = std::min(std::abs(vyB*getdt()),cell->getHeight());
				} else {
					dir = DOWN;
					nextCell = cellB;
					dist = std::abs(vyB*getdt());
				}

				face = cell->getWidth();
				massFluxB = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass();}, dir, 0, false, false);
				EFluxB = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getE(false)*c->getSize();}, dir, 0, false, false, [](FluidCell *c, long double m){c->newE -= m/c->getSize();});
				vxFluxB = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass()*c->getVelocity().getVx();}, dir, 0);
				vyFluxB = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass()*c->getVelocity().getVy();}, dir, 0);
				XFluxB = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getX()*c->getMass();}, dir, 0);
				YFluxB = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getY()*c->getMass();}, dir, 0);
				ZFluxB = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getZ()*c->getMass();}, dir, 0);
				unFluxB = getFlux2(nextCell, dist, face, [](FluidCell *c) {return 1;}, dir, 0);
			}
			if (vyB < 0) {
				massFluxB *= -1;
				EFluxB *= -1;
				vxFluxB *= -1;
				vyFluxB *= -1;
				XFluxB *= -1;
				YFluxB *= -1;
				ZFluxB *= -1;
				unFluxB *= -1;
			}
		}

		if (cell == cellR) {
			massFluxR = std::max((long double) 0, massFluxL);
			EFluxR = std::max((long double) 0, EFluxL);
			vxFluxR = std::max((long double) 0, vxFluxL);
			vyFluxR = std::max((long double) 0, vyFluxL);
			XFluxR = std::max((long double) 0, XFluxL);
			YFluxR = std::max((long double) 0, YFluxL);
			ZFluxR = std::max((long double) 0, ZFluxL);
		}

		if (cell == cellL) {
			massFluxL = std::min((long double) 0, massFluxR);
			EFluxL = std::min((long double) 0, EFluxR);
			vxFluxL = std::min((long double) 0, vxFluxR);
			vyFluxL = std::min((long double) 0, vyFluxR);
			XFluxL = std::min((long double) 0, XFluxR);
			YFluxL = std::min((long double) 0, YFluxR);
			ZFluxL = std::min((long double) 0, ZFluxR);
		}

		if (cell == cellT) {
			massFluxT = std::max((long double) 0, massFluxB);
			EFluxT = std::max((long double) 0, EFluxB);
			vxFluxT = std::max((long double) 0, vxFluxB);
			vyFluxT = std::max((long double) 0, vyFluxB);
			XFluxT = std::max((long double) 0, XFluxB);
			YFluxT = std::max((long double) 0, YFluxB);
			ZFluxT = std::max((long double) 0, ZFluxB);
		}

		if (cell == cellB) {
			massFluxB = std::min((long double) 0, massFluxT);
			EFluxB = std::min((long double) 0, EFluxT);
			vxFluxB = std::min((long double) 0, vxFluxT);
			vyFluxB = std::min((long double) 0, vyFluxT);
			XFluxB = std::min((long double) 0, XFluxT);
			YFluxB = std::min((long double) 0, YFluxT);
			ZFluxB = std::min((long double) 0, ZFluxT);
		}

		if (cell == cellR || cell == cellL || cell == cellT || cell == cellB) {
			vxFluxR = 0;
			vxFluxL = 0;
			vxFluxT = 0;
			vxFluxB = 0;
			vyFluxR = 0;
			vyFluxL = 0;
			vyFluxT = 0;
			vyFluxB = 0;
		}

		float unX = 1 - unFluxR + unFluxL;
		float unY = 1 - unFluxT + unFluxB;

		long double massFluxes[4] = {massFluxR, -massFluxL, massFluxT, -massFluxB};
		long double EFluxes[4] = {EFluxR, -EFluxL, EFluxT, -EFluxB};

		long double massFluxOut = 0;
		long double EFluxOut = 0;
		for (int k = 0; k < 4; k++) {
			if (massFluxes[k] > 0) {
				massFluxOut += massFluxes[k];
				EFluxOut += EFluxes[k];
			}
		}

		cell->newMass -= massFluxOut;
		cell->newE -= EFluxOut/cell->getSize();

		cell->newMassx = cell->getMass() + 0.5*(-massFluxR + massFluxL);
		cell->newMassy = cell->getMass() + 0.5*(-massFluxT + massFluxB);

		cell->newEx = std::max((long double) 0.0f, (E*cell->getSize() + 0.5*(-EFluxR + EFluxL))/cell->getSize());
		cell->newEy = std::max((long double) 0.0f, (E*cell->getSize() + 0.5*(-EFluxT + EFluxB))/cell->getSize());

		double newVx = (mass*cell->getVelocity().getVx() + -vxFluxR + vxFluxL + -vxFluxT + vxFluxB);
		double newVxx = (mass*cell->getVelocity().getVx() + 0.5*(-vxFluxR + vxFluxL))/cell->newMassx;
		double newVxy = (mass*cell->getVelocity().getVx() + 0.5*(-vxFluxT + vxFluxB))/cell->newMassy;
		double newVy = (mass*cell->getVelocity().getVy() + -vyFluxR + vyFluxL + -vyFluxT + vyFluxB);
		double newVyx = (mass*cell->getVelocity().getVy() + 0.5*(-vyFluxR + vyFluxL))/cell->newMassx;
		double newVyy = (mass*cell->getVelocity().getVy() + 0.5*(-vyFluxT + vyFluxB))/cell->newMassy;

		cell->newVelocityx = VelocityVector(newVxx, newVyx);
		cell->newVelocityy = VelocityVector(newVxy, newVyy);
		cell->newVelocity = VelocityVector(newVx, newVy);

		cell->newXx = std::min(1.0f,std::max(0.0f,(float)((mass*X + 0.5*(-XFluxR + XFluxL))/cell->newMassx)) );
		cell->newXy = std::min(1.0f,std::max(0.0f,(float)((mass*X + 0.5*(-XFluxT + XFluxB))/cell->newMassy)) );

		cell->newYx = std::min(1.0f,std::max(0.0f,(float)((mass*Y + 0.5*(-YFluxR + YFluxL))/cell->newMassx)) );
		cell->newYy = std::min(1.0f,std::max(0.0f,(float)((mass*Y + 0.5*(-YFluxT + YFluxB))/cell->newMassy)) );

		cell->newZx = std::min(1.0f,std::max(0.0f,(float)((mass*Z + 0.5*(-ZFluxR + ZFluxL))/cell->newMassx)) );
		cell->newZy = std::min(1.0f,std::max(0.0f,(float)((mass*Z + 0.5*(-ZFluxT + ZFluxB))/cell->newMassy)) );

	}
}

__global__ void advect_kernel2(std::vector<FluidCell*> leaves) {
	int workIdx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (workIdx < leaves.size()) {
		FluidCell *cell = leaves[workIdx];
		long double mass = cell->getMass();
		long double E = cell->getE(false);

		long double massFluxOut = abs(cell->newMass);
		long double EFluxOut = abs(cell->newE*cell->getSize());
		
		cell->massODs = std::min((long double) 1.0f, mass/massFluxOut);
		cell->EODs = std::min((long double) 1.0f, E*cell->getSize()/EFluxOut);

		cell->newMass = 0;
		cell->newE = 0;
	}
}

__global__ void advect_kernel3(std::vector<FluidCell*> leaves, bool& minSizeSmall, long double& thisMaxDensity, long double& maxDensity) {
	int workIdx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (workIdx < leaves.size()) {
		FluidCell *cell = leaves[workIdx];
		FluidCell *cellR = cell->getRight();
		FluidCell *cellL = cell->getLeft();
		FluidCell *cellT = cell->getTop();
		FluidCell *cellB = cell->getBottom();
		VelocityVector vC = cell->getVelocity();
		float X = cell->getX();
		float Y = cell->getY();
		float Z = cell->getZ();

		if (cell->getHeight() == minSize || cell->getWidth() == minSize) minSizeSmall = false;

		long double massFluxR = 0;
		long double massFluxL = 0;
		long double massFluxT = 0;
		long double massFluxB = 0;
		long double EFluxR = 0;
		long double EFluxL = 0;
		long double EFluxT = 0;
		long double EFluxB = 0;
		long double vxFluxR = 0;
		long double vyFluxR = 0;
		long double vxFluxL = 0;
		long double vyFluxL = 0;
		long double vxFluxT = 0;
		long double vyFluxT = 0;
		long double vxFluxB = 0;
		long double vyFluxB = 0;
		long double XFluxR = 0;
		long double XFluxL = 0;
		long double XFluxT = 0;
		long double XFluxB = 0;
		long double YFluxR = 0;
		long double YFluxL = 0;
		long double YFluxT = 0;
		long double YFluxB = 0;
		long double ZFluxR = 0;
		long double ZFluxL = 0;
		long double ZFluxT = 0;
		long double ZFluxB = 0;
		long double vxR;
		long double vxL;
		long double vyT;
		long double vyB;
		long double massR, massL, massT, massB;
		long double ER, EL, ET, EB;
		long double mass = cell->getMass();
		long double E = cell->getE(false);
		float XR, XL, XT, XB, YR, YL, YT, YB, ZR, ZL, ZT, ZB;
		long double dist, face;

		if (cell->getDensity() > thisMaxDensity) thisMaxDensity = cell->getDensity();
		vxR = (cell->getRight()->getVelocity().getVx() + vC.getVx())/2;
		vxL = (cell->getLeft()->getVelocity().getVx() + vC.getVx())/2;
		vyT = (cell->getTop()->getVelocity().getVy() + vC.getVy())/2;
		vyB = (cell->getBottom()->getVelocity().getVy() + vC.getVy())/2;
		XR = cellR->getX();
		XL = cellL->getX();
		XT = cellT->getX();
		XB = cellB->getX();
		YR = cellR->getY();
		YL = cellL->getY();
		YT = cellT->getY();
		YB = cellB->getY();
		ZR = cellR->getZ();
		ZL = cellL->getZ();
		ZT = cellT->getZ();
		ZB = cellB->getZ();
		Direction dir;
		FluidCell *nextCell;

		// not a boundary
		if (cellR != cell) {
			if (cellR->hasChildren()) {
				FluidCell *children[2] = {cellR->nw, cellR->sw};
				for (int i = 0; i < 2; i++) {
					vxR = (children[i]->getVelocity().getVx() + vC.getVx())/2;
					massR = children[i]->getMass();
					ER = children[i]->getE(false);
					if (vxR < 0) {
						dir = RIGHT;
						nextCell = children[i];
						dist = std::abs(vxR*getdt());
					} else {
						dir = LEFT;
						nextCell = cell;
						dist = std::min(std::abs(vxR*getdt()),cell->getWidth());
					}

					face = children[i]->getHeight();
					massFluxR += getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newMassy*c->massODs;}, dir, 0, false, false, [](FluidCell *c, long double m){c->newMass -= m;});
					EFluxR += getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newEy*c->getSize()*c->EODs;}, dir, 0, false, false, [](FluidCell *c, long double m){c->newE -= m/c->getSize();});
					vxFluxR += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->newMassy*c->newVelocityy.getVx()*c->massODs;}, dir, 0);
					vyFluxR += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->newMassy*c->newVelocityy.getVy()*c->massODs;}, dir, 0);
					XFluxR += getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newXy*c->newMassy*c->massODs;}, dir, 0);
					YFluxR += getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newYy*c->newMassy*c->massODs;}, dir, 0);
					ZFluxR += getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newZy*c->newMassy*c->massODs;}, dir, 0);
				}
				
			} else {
				if (vxR < 0) {
					dir = RIGHT;
					nextCell = cellR;
					dist = std::abs(vxR*getdt());
				} else {
					dir = LEFT;
					nextCell = cell;
					dist = std::min(std::abs(vxR*getdt()),cell->getWidth());
				}

				face = cell->getHeight();
				massFluxR = getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newMassy*c->massODs;}, dir, 0, false, false, [](FluidCell *c, long double m){c->newMass -= m;});
				EFluxR = getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newEy*c->getSize()*c->EODs;}, dir, 0, false, false, [](FluidCell *c, long double m){c->newE -= m/c->getSize();});
				vxFluxR = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->newMassy*c->newVelocityy.getVx()*c->massODs;}, dir, 0);
				vyFluxR = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->newMassy*c->newVelocityy.getVy()*c->massODs;}, dir, 0);
				XFluxR = getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newXy*c->newMassy*c->massODs;}, dir, 0);
				YFluxR = getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newYy*c->newMassy*c->massODs;}, dir, 0);
				ZFluxR = getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newZy*c->newMassy*c->massODs;}, dir, 0);
			}

			if (vxR < 0) {
				massFluxR *= -1;
				EFluxR *= -1;
				vxFluxR *= -1;
				vyFluxR *= -1;
				XFluxR *= -1;
				YFluxR *= -1;
				ZFluxR *= -1;
			}
		}

		// not a boundary
		if (cellL != cell) {
			if (cellL->hasChildren()) {
				FluidCell *children[2] = {cellL->ne, cellL->se};
				for (int i = 0; i < 2; i++) {
					vxL = (children[i]->getVelocity().getVx() + vC.getVx())/2;
					massL = children[i]->getMass();
					EL = children[i]->getE(false);
					if (vxL < 0) {
						dir = RIGHT;
						nextCell = cell;
						dist = std::min(std::abs(vxL*getdt()),cell->getWidth());
					} else {
						dir = LEFT;
						nextCell = children[i];
						dist = std::abs(vxL*getdt());
					}

					face = children[i]->getHeight();
					massFluxL += getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newMassy*c->massODs;}, dir, 0, false, false, [](FluidCell *c, long double m){c->newMass -= m;});
					EFluxL += getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newEy*c->getSize()*c->EODs;}, dir, 0, false, false, [](FluidCell *c, long double m){c->newE -= m/c->getSize();});
					vxFluxL += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->newMassy*c->newVelocityy.getVx()*c->massODs;}, dir, 0);
					vyFluxL += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->newMassy*c->newVelocityy.getVy()*c->massODs;}, dir, 0);
					XFluxL += getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newXy*c->newMassy*c->massODs;}, dir, 0);
					YFluxL += getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newYy*c->newMassy*c->massODs;}, dir, 0);
					ZFluxL += getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newZy*c->newMassy*c->massODs;}, dir, 0);
				}
			} else {
				massL = cellL->getMass();
				EL = cellL->getE(false);
				if (vxL < 0) {
					dir = RIGHT;
					nextCell = cell;
					dist = std::min(std::abs(vxL*getdt()),cell->getWidth());
				} else {
					dir = LEFT;
					nextCell = cellL;
					dist = std::abs(vxL*getdt());
				}

				face = cell->getHeight();
				massFluxL = getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newMassy*c->massODs;}, dir, 0, false, false, [](FluidCell *c, long double m){c->newMass -= m;});
				EFluxL = getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newEy*c->getSize()*c->EODs;}, dir, 0, false, false, [](FluidCell *c, long double m){c->newE -= m/c->getSize();});
				vxFluxL = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->newMassy*c->newVelocityy.getVx()*c->massODs;}, dir, 0);
				vyFluxL = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->newMassy*c->newVelocityy.getVy()*c->massODs;}, dir, 0);
				XFluxL = getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newXy*c->newMassy*c->massODs;}, dir, 0);
				YFluxL = getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newYy*c->newMassy*c->massODs;}, dir, 0);
				ZFluxL = getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newZy*c->newMassy*c->massODs;}, dir, 0);
			}
			if (vxL < 0) {
				massFluxL *= -1;
				EFluxL *= -1;
				vxFluxL *= -1;
				vyFluxL *= -1;
				XFluxL *= -1;
				YFluxL *= -1;
				ZFluxL *= -1;
			}
		}

		// not a boundary
		if (cellT != cell) {
			if (cellT->hasChildren()) {
				FluidCell *children[2] = {cellT->sw, cellT->se};
				for (int i = 0; i < 2; i++) {
					vyT = (children[i]->getVelocity().getVy() + vC.getVy())/2;
					massT = children[i]->getMass();
					ET = children[i]->getE(false);
					if (vyT < 0) {
						dir = UP;
						nextCell = children[i];
						dist = std::abs(vyT*getdt());
					} else {
						dir = DOWN;
						nextCell = cell;
						dist = std::min(std::abs(vyT*getdt()),cell->getHeight());
					}

					face = children[i]->getWidth();
					massFluxT += getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newMassx*c->massODs;}, dir, 0, false, false, [](FluidCell *c, long double m){c->newMass -= m;});
					EFluxT += getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newEx*c->getSize()*c->EODs;}, dir, 0, false, false, [](FluidCell *c, long double m){c->newE -= m/c->getSize();});
					vxFluxT += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->newMassx*c->newVelocityx.getVx()*c->massODs;}, dir, 0);
					vyFluxT += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->newMassx*c->newVelocityx.getVy()*c->massODs;}, dir, 0);
					XFluxT += getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newXx*c->newMassx*c->massODs;}, dir, 0);
					YFluxT += getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newYx*c->newMassx*c->massODs;}, dir, 0);
					ZFluxT += getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newZx*c->newMassx*c->massODs;}, dir, 0);
				}
				
			} else {
				massT = cellT->getMass();
				ET = cellT->getE(false);
				if (vyT < 0) {
					dir = UP;
					nextCell = cellT;
					dist = std::abs(vyT*getdt());
				} else {
					dir = DOWN;
					nextCell = cell;
					dist = std::min(std::abs(vyT*getdt()),cell->getHeight());
				}

				face = cell->getWidth();
				massFluxT = getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newMassx*c->massODs;}, dir, 0, false, false, [](FluidCell *c, long double m){c->newMass -= m;});
				EFluxT = getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newEx*c->getSize()*c->EODs;}, dir, 0, false, false, [](FluidCell *c, long double m){c->newE -= m/c->getSize();});
				vxFluxT = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->newMassx*c->newVelocityx.getVx()*c->massODs;}, dir, 0);
				vyFluxT = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->newMassx*c->newVelocityx.getVy()*c->massODs;}, dir, 0);
				XFluxT = getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newXx*c->newMassx*c->massODs;}, dir, 0);
				YFluxT = getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newYx*c->newMassx*c->massODs;}, dir, 0);
				ZFluxT = getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newZx*c->newMassx*c->massODs;}, dir, 0);
			}
			if (vyT < 0) {
				massFluxT *= -1;
				EFluxT *= -1;
				vxFluxT *= -1;
				vyFluxT *= -1;
				XFluxT *= -1;
				YFluxT *= -1;
				ZFluxT *= -1;
			}
		}

		// not a boundary
		if (cellB != cell) {
			if (cellB->hasChildren()) {
				FluidCell *children[2] = {cellB->nw, cellB->ne};
				for (int i = 0; i < 2; i++) {
					vyB = (children[i]->getVelocity().getVy() + vC.getVy())/2;
					massB = children[i]->getMass();
					EB = children[i]->getE(false);
					if (vyB < 0) {
						dir = UP;
						nextCell = cell;
						dist = std::min(std::abs(vyB*getdt()),cell->getHeight());
					} else {
						dir = DOWN;
						nextCell = children[i];
						dist = std::abs(vyB*getdt());
					}
					face = children[i]->getWidth();
					massFluxB += getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newMassx*c->massODs;}, dir, 0, false, false, [](FluidCell *c, long double m){c->newMass -= m;});
					EFluxB += getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newEx*c->getSize()*c->EODs;}, dir, 0, false, false, [](FluidCell *c, long double m){c->newE -= m/c->getSize();});

					vxFluxB += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->newMassx*c->newVelocityx.getVx()*c->massODs;}, dir, 0);
					vyFluxB += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->newMassx*c->newVelocityx.getVy()*c->massODs;}, dir, 0);
					XFluxB += getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newXx*c->newMassx*c->massODs;}, dir, 0);
					YFluxB += getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newYx*c->newMassx*c->massODs;}, dir, 0);
					ZFluxB += getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newZx*c->newMassx*c->massODs;}, dir, 0);
				}
				
			} else {
				massB = cellB->getMass();
				EB = cellB->getE(false);
				if (vyB < 0) {
					dir = UP;
					nextCell = cell;
					dist = std::min(std::abs(vyB*getdt()),cell->getHeight());
				} else {
					dir = DOWN;
					nextCell = cellB;
					dist = std::abs(vyB*getdt());
				}
				face = cell->getWidth();
				massFluxB = getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newMassx*c->massODs;}, dir, 0, false, false, [](FluidCell *c, long double m){c->newMass -= m;});
				EFluxB = getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newEx*c->getSize()*c->EODs;}, dir, 0, false, false, [](FluidCell *c, long double m){c->newE -= m/c->getSize();});

				vxFluxB = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->newMassx*c->newVelocityx.getVx()*c->massODs;}, dir, 0);
				vyFluxB = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->newMassx*c->newVelocityx.getVy()*c->massODs;}, dir, 0);
				XFluxB = getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newXx*c->newMassx*c->massODs;}, dir, 0);
				YFluxB = getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newYx*c->newMassx*c->massODs;}, dir, 0);
				ZFluxB = getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newZx*c->newMassx*c->massODs;}, dir, 0);
			}
			if (vyB < 0) {
				massFluxB *= -1;
				EFluxB *= -1;
				vxFluxB *= -1;
				vyFluxB *= -1;
				XFluxB *= -1;
				YFluxB *= -1;
				ZFluxB *= -1;
			}
		}

		if (cell == cellR) {
			massFluxR = std::max((long double) 0, massFluxL);
			EFluxR = std::max((long double) 0, EFluxL);
			vxFluxR = std::max((long double) 0, vxFluxL);
			vyFluxR = std::max((long double) 0, vyFluxL);
			XFluxR = std::max((long double) 0, XFluxL);
			YFluxR = std::max((long double) 0, YFluxL);
			ZFluxR = std::max((long double) 0, ZFluxL);
		}

		if (cell == cellL) {
			massFluxL = std::min((long double) 0, massFluxR);
			EFluxL = std::min((long double) 0, EFluxR);
			vxFluxL = std::min((long double) 0, vxFluxR);
			vyFluxL = std::min((long double) 0, vyFluxR);
			XFluxL = std::min((long double) 0, XFluxR);
			YFluxL = std::min((long double) 0, YFluxR);
			ZFluxL = std::min((long double) 0, ZFluxR);
		}

		if (cell == cellT) {
			massFluxT = std::max((long double) 0, massFluxB);
			EFluxT = std::max((long double) 0, EFluxB);
			vxFluxT = std::max((long double) 0, vxFluxB);
			vyFluxT = std::max((long double) 0, vyFluxB);
			XFluxT = std::max((long double) 0, XFluxB);
			YFluxT = std::max((long double) 0, YFluxB);
			ZFluxT = std::max((long double) 0, ZFluxB);
		}

		if (cell == cellB) {
			massFluxB = std::min((long double) 0, massFluxT);
			EFluxB = std::min((long double) 0, EFluxT);
			vxFluxB = std::min((long double) 0, vxFluxT);
			vyFluxB = std::min((long double) 0, vyFluxT);
			XFluxB = std::min((long double) 0, XFluxT);
			YFluxB = std::min((long double) 0, YFluxT);
			ZFluxB = std::min((long double) 0, ZFluxT);
		}

		if (cell == cellR || cell == cellL || cell == cellT || cell == cellB) {
			vxFluxR = 0;
			vxFluxL = 0;
			vxFluxT = 0;
			vxFluxB = 0;
			vyFluxR = 0;
			vyFluxL = 0;
			vyFluxT = 0;
			vyFluxB = 0;
		}

		// newMass and newE might have been subtracted from by "far" cells
		cell->newMass += (cell->getMass() - massFluxR + massFluxL - massFluxT + massFluxB);

		cell->newE += (cell->getE(false)*cell->getSize() - EFluxR + EFluxL - EFluxT + EFluxB)/cell->getSize();
	
		// save dividing by newMass for last loop in case newMass gets updated
		double newVx = (mass*cell->getVelocity().getVx() + -vxFluxR + vxFluxL + -vxFluxT + vxFluxB);
		double newVy = (mass*cell->getVelocity().getVy() + -vyFluxR + vyFluxL + -vyFluxT + vyFluxB);

		cell->newX = std::max((long double) 0.0f,((mass*X - XFluxR + XFluxL - XFluxT + XFluxB)));
		cell->newY = std::max((long double) 0.0f,((mass*Y - YFluxR + YFluxL - YFluxT + YFluxB)));
		cell->newZ = std::max((long double) 0.0f,((mass*Z - ZFluxR + ZFluxL - ZFluxT + ZFluxB)));
	}
}

__global__ void advect_kernel4(std::vector<FluidCell*> leaves, long double& totMass) {
	int workIdx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (workIdx < leaves.size()) {
		FluidCell *cell = leaves[workIdx];
		if (std::isfinite(cell->newMass+cell->newE+cell->newVelocity.getMag()+cell->newX+cell->newY+cell->newZ)) {
			if (cell->newMass == 0) {
				cell->newVelocity = VelocityVector(0,0);
				cell->newE = 0;
				cell->newX = 0;
				cell->newY = 0;
				cell->newZ = 0;
			}

			cell->setMass(cell->newMass);
			cell->setE(cell->newE);
			totMass += std::max((long double) 0.0f, cell->newMass);
			cell->setVelocity(cell->newVelocity/cell->newMass);
			long double factor = cell->newMass/(cell->newX+cell->newY+cell->newZ);

			cell->setX(std::max(0.0f,std::min(1.0f,(float) (factor*cell->newX/cell->newMass))));
			cell->setY(std::max(0.0f,std::min(1.0f,(float) (factor*cell->newY/cell->newMass))));
			cell->setZ(std::max(0.0f,std::min(1.0f,(float) (factor*cell->newZ/cell->newMass))));
			cell->sete(cell->getE(false) - 0.5*cell->getDensity()*cell->getVelocity().getMag()*cell->getVelocity().getMag());
			cell->getTemp(true);
			cell->getPressure(true);
		}

		cell->massODs = 1;
		cell->EODs = 1;
		cell->newMass = 0;
		cell->newE = 0;
		cell->newX = 0;
		cell->newY = 0;
		cell->newZ = 0;
		cell->newVelocity = VelocityVector(0,0);
	}
}

long double getFlux2(FluidCell *cell, long double dist, long double boundary, std::function<long double(FluidCell *)> func, Direction dir, long double totFlux, bool forceFarOff=true, bool farCell=false, std::function<void (FluidCell *, long double)> func2=[](FluidCell *c, long double m){c->newMass -= m;}) {
	if (!std::isfinite(dist)) return 0;
	long double flux = 0;
	long double leng = (dir == LEFT) || (dir == RIGHT) ? cell->getWidth() : cell->getHeight();
	if (cell->hasChildren()) {
		switch (dir) {
			case RIGHT:
				flux = getFlux2(cell->nw, dist, boundary, func, dir, totFlux) + getFlux2(cell->sw, dist, boundary, func, dir, totFlux, forceFarOff, farCell, func2);
				// return getFlux2(cell->nw, dist, boundary, func, dir, totFlux) + getFlux2(cell->sw, dist, boundary, func, dir, totFlux);
				break;
			case LEFT:
				flux = getFlux2(cell->ne, dist, boundary, func, dir, totFlux) + getFlux2(cell->se, dist, boundary, func, dir, totFlux, forceFarOff, farCell, func2);
				// return getFlux2(cell->ne, dist, boundary, func, dir, totFlux) + getFlux2(cell->se, dist, boundary, func, dir, totFlux);
				break;
			case UP:
				flux = getFlux2(cell->se, dist, boundary, func, dir, totFlux) + getFlux2(cell->sw, dist, boundary, func, dir, totFlux, forceFarOff, farCell, func2);
				// return getFlux2(cell->se, dist, boundary, func, dir, totFlux) + getFlux2(cell->sw, dist, boundary, func, dir, totFlux);
				break;
			case DOWN:
				flux = getFlux2(cell->ne, dist, boundary, func, dir, totFlux) + getFlux2(cell->nw, dist, boundary, func, dir, totFlux, forceFarOff, farCell, func2);
				// return getFlux2(cell->ne, dist, boundary, func, dir, totFlux) + getFlux2(cell->nw, dist, boundary, func, dir, totFlux);
		}
	} else if (dist <= leng) {
		flux = totFlux + func(cell)*boundary*dist/cell->getSize();
		// return totFlux + func(cell)*boundary*dist/cell->getSize();
	} else {
		FluidCell *nextCell;
		switch (dir) {
			case RIGHT:
				nextCell = cell->getRight();
				break;
			case LEFT:
				nextCell = cell->getLeft();
				break;
			case UP:
				nextCell = cell->getTop();
				break;
			case DOWN:
				nextCell = cell->getBottom();
		}
		if (nextCell != cell) {
			flux = getFlux2(nextCell, dist-leng, boundary, func, dir, totFlux + func(cell)*boundary*leng/cell->getSize(), forceFarOff, true, func2);
			// return getFlux2(nextCell, dist-leng, boundary, func, dir, totFlux + func(cell)*boundary*leng/cell->getSize(), true);
		} else {
			flux = totFlux + func(cell)*boundary*leng/cell->getSize();
			// return totFlux + func(cell)*boundary*leng/cell->getSize();
		}
	}
	if (!forceFarOff && farCell) {
		// std::cout << "far out dude\n";
		func2(cell,flux-totFlux);
	}

	return flux;
}


class Simulator {
    public:
        Simulator(double width, double height, int startDepth, bool display=true, bool save=false, std::string fn=""): width(width), height(height), display(display), save(save), filename(fn) {
            dt = 0.1;
            // in meters per pixel
            SCALE_H = height / consts::GRID_HEIGHT;
            SCALE_W = width / consts::GRID_WIDTH;
            cudaMallocManaged(&grid, sizeof(FluidGrid));
            grid[0] = FluidGrid(width, height, startDepth, dt);
            output.open(filename);
            SDL_Init(SDL_INIT_VIDEO);       // Initializing SDL as Video
            SDL_CreateWindowAndRenderer(consts::GRID_WIDTH, consts::GRID_HEIGHT, 0, &window, &renderer);
            SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0);      // setting draw color
            SDL_RenderClear(renderer);

        }

        void drawCells() {
            TreeLoc m;
            uint32_t depth, index;
            xyPos xyBL;
            double cellWidth, cellHeight;
            double thisMaxPressure = 0;
            double thisMaxMass = 0;
            double thisMinDensity = maxDensity;
            double thisMaxDensity = 0;
            double thisMaxTemperature = 0;
            double thisMinTemperature = 0;
            double thisMinGP = 0;
            double thisMaxe = 0;
            double thisMaxFusion = 0;
	    double thisMaxPG = 0;
            SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
            SDL_RenderClear(renderer);
            std::vector<FluidCell *> leaves = grid->getLeafCells();
            for (int i = 0; i < leaves.size(); i++) {

                FluidCell *cell = leaves[i];
                depth = cell->depth;
                index = cell->index;
                double pressure = cell->getPressure(false);
                double density = cell->getDensity();
                double temperature = cell->getTemp(false);
                double gravPotential = cell->getGravPotential();
                double e = cell->gete(false);
                float X = std::max(0.0f,std::min(1.0f,cell->getX()));
                float Y = std::max(0.0f,std::min(1.0f,cell->getY()));
                float Z = std::max(0.0f,std::min(1.0f,cell->getZ()));
                float fusion = cell->getFusionEnergy();
                uint degenerate = cell->getDegenerate();
                xyBL = grid->getXY(depth, index);
                cellWidth = width / pow(pow(4,depth), 0.5);
                cellHeight = height / pow(pow(4,depth), 0.5);

                SDL_Rect rect{(int)(xyBL.x/SCALE_W), (int)(consts::GRID_HEIGHT - ((xyBL.y+cellHeight)/SCALE_H)), (int)(cellWidth/SCALE_W)+1, (int)(cellHeight/SCALE_H)+1};
                if (density > thisMaxDensity) thisMaxDensity = density;
                if (density < thisMinDensity) thisMinDensity = density;
                if (pressure > thisMaxPressure) {
                    thisMaxPressure = pressure;
                }
                if (temperature > thisMaxTemperature) {
                    // && density > 0.01*maxDensity
                    thisMaxTemperature = temperature;
                }
                if (gravPotential < thisMinGP) thisMinGP = gravPotential;
                if (e > thisMaxe) thisMaxe = e;
                if (fusion > thisMaxFusion) thisMaxFusion = fusion;
				if (cell->PG > thisMaxPG) thisMaxPG = cell->PG;

                double maxBit = 255.0;
                double zero = 0;
                if (pressureDisplay) {
                    double scaledP_R = std::min(maxBit,255 * (pressure)/(maxPressure));
                    if (scaledP_R < 0) scaledP_R = 0;

                    SDL_SetRenderDrawColor(renderer, scaledP_R, 0, 255, 255);
                } else if (densityDisplay) {
                    double scaled_dens = std::sqrt(density/maxDensity) * 255;
                    // scaled_dens = std::log10(cell->getMass())/std::log10(maxDensity*cell->getSize()) * 255;
                    if (scaled_dens > 255) scaled_dens = 255;

                    SDL_SetRenderDrawColor(renderer, scaled_dens*(1-Z), scaled_dens*std::max(0.0f,(1-Y-Z)), scaled_dens*Z, 255);
                    // SDL_RenderFillRect(renderer, &rect);
                } else if (temperatureDisplay) {
                    double scaled_temp = std::min(maxBit,temperature/maxTemperature * 255);
                    // if (density < 0.01*maxDensity) scaled_temp = 0;
                    SDL_SetRenderDrawColor(renderer, 0, scaled_temp, 0, 255);
                } else if (gravPotentialDisplay) {
                    double scaled_pot = std::max(zero,std::min(maxBit,gravPotential/minGP * 255));
                    // if (i == rows/2 && j == cols/2) printf("scaledGP:%f\n",gravPotential/minGP);
                    SDL_SetRenderDrawColor(renderer, 255, scaled_pot, 0, 255);

                } else if (energyDisplay) {
                    double scaled_e = std::max(zero,std::min(maxBit,e/maxe * 255));
                    SDL_SetRenderDrawColor(renderer, scaled_e, 0, 0, 255);
                } else if (hydrogenDisplay) {
                    int scaled_X = std::min(255.0f,X*255);
                    // std::cout << X << std::endl;
                    SDL_SetRenderDrawColor(renderer, scaled_X, scaled_X, 0, 255);
                } else if (heliumDisplay) {
                    int scaled_Y = std::min(255.0f,Y*255);
                    SDL_SetRenderDrawColor(renderer, scaled_Y, 0, scaled_Y, 255);
                } else if (ZDisplay) {
                    int scaled_Z = std::min(255.0f,Z*255);
                    SDL_SetRenderDrawColor(renderer, 0, 0, scaled_Z, 255);
                } else if (fusionDisplay) {
                    double scaled_fusion = std::max(zero,std::min(maxBit,fusion/maxFusion * 255));
                    if (maxFusion == 0) scaled_fusion = 0;
                    SDL_SetRenderDrawColor(renderer, 0, scaled_fusion, scaled_fusion, 255);
                } else if (degenerateDisplay) {
                    double deg = degenerate*255;
		    deg = std::max(zero,std::min(maxBit,(double) cell->PG/maxPG * 255));
                    SDL_SetRenderDrawColor(renderer, 0, 0, deg, 255);
                }

                SDL_RenderFillRect(renderer, &rect);
                if (gridDisplay) {
                    SDL_SetRenderDrawColor(renderer, 255,255,255, 255); // White outline
                    SDL_RenderDrawRect(renderer, &rect);
                }

            }
            if (velocityDisplay) {
                for (int i = 0; i < leaves.size(); i++) {

                    FluidCell *cell = leaves[i];
                    depth = cell->depth;
                    index = cell->index;
                    xyBL = grid->getXY(depth, index);
                    cellWidth = width / pow(pow(4,depth), 0.5);
                    cellHeight = height / pow(pow(4,depth), 0.5);

                    VelocityVector v = cell->getVelocity();
                    Arrow arrow{(int)((xyBL.x+cellWidth/2)/SCALE_W), (int)(consts::GRID_HEIGHT - ((xyBL.y+cellHeight/2)/SCALE_H)), 2*(int)(v.getVx()*grid->getdt()/cellWidth * cellWidth/SCALE_W), -2*(int)(v.getVy()*grid->getdt()/cellHeight * cellHeight/SCALE_H)};
                    SDL_SetRenderDrawColor(renderer, 255,255,255, 255);
                    if (std::isfinite(v.getMag()) && v.getMag() > 0) drawArrow(renderer, arrow);

                }
            }

			maxPG = thisMaxPG;
            maxPressure = thisMaxPressure;
            maxDensity = thisMaxDensity;
            minDensity = thisMinDensity;
            // maxDensity = 10;
            maxMass = thisMaxMass;
            maxTemperature = thisMaxTemperature;
            minTemperature = thisMinTemperature;
            // std::cout << maxTemperature << std::endl;
            minGP = thisMinGP; // grav potential
            maxe = thisMaxe;
            maxFusion = thisMaxFusion;
            SDL_RenderPresent(renderer);
        }

        void step(SDL_Event event) {
            grid->update();
            dt = grid->getdt();
            t += grid->getdt();
            if (event.key.type == SDL_KEYDOWN) {
                switch (event.key.keysym.sym) {
                    case SDLK_p:
                        densityDisplay = 0;
                        temperatureDisplay = 0;
                        gravPotentialDisplay = 0;
                        energyDisplay = 0;
                        pressureDisplay = !(pressureDisplay);
                        hydrogenDisplay = 0;
                        heliumDisplay = 0;
                        ZDisplay = 0;
                        fusionDisplay = 0;
                        degenerateDisplay = 0;
                        break;
                    case SDLK_d:
                        pressureDisplay = 0;
                        temperatureDisplay = 0;
                        gravPotentialDisplay = 0;
                        energyDisplay = 0;
                        densityDisplay = !(densityDisplay);
                        hydrogenDisplay = 0;
                        heliumDisplay = 0;
                        ZDisplay = 0;
                        fusionDisplay = 0;
                        degenerateDisplay = 0;
                        std::cout << "max Density: " << maxDensity << " min Density: " << minDensity << std::endl;
                        break;
                    case SDLK_t:
                        densityDisplay = 0;
                        pressureDisplay = 0;
                        gravPotentialDisplay = 0;
                        energyDisplay = 0;
                        temperatureDisplay = !(temperatureDisplay);
                        hydrogenDisplay = 0;
                        heliumDisplay = 0;
                        ZDisplay = 0;
                        fusionDisplay = 0;
                        degenerateDisplay = 0;
                        std::cout << "max Temperature: " << maxTemperature << std::endl;
                        break;
                    case SDLK_u:
                        densityDisplay = 0;
                        pressureDisplay = 0;
                        temperatureDisplay = 0;
                        energyDisplay = 0;
                        gravPotentialDisplay = !(gravPotentialDisplay);
                        hydrogenDisplay = 0;
                        heliumDisplay = 0;
                        ZDisplay = 0;
                        fusionDisplay = 0;
                        degenerateDisplay = 0;
                        break;
                    case SDLK_e:
                        densityDisplay = 0;
                        pressureDisplay = 0;
                        temperatureDisplay = 0;
                        gravPotentialDisplay = 0;
                        energyDisplay = !(energyDisplay);
                        hydrogenDisplay = 0;
                        heliumDisplay = 0;
                        ZDisplay = 0;
                        fusionDisplay = 0;
                        degenerateDisplay = 0;
                        std::cout << "max internal energy: " << maxe << std::endl;
                        break;
                    case SDLK_7:
                        densityDisplay = 0;
                        pressureDisplay = 0;
                        temperatureDisplay = 0;
                        gravPotentialDisplay = 0;
                        energyDisplay = 0;
                        hydrogenDisplay = !(hydrogenDisplay);
                        heliumDisplay = 0;
                        ZDisplay = 0;
                        fusionDisplay = 0;
                        degenerateDisplay = 0;
                        break;
                    case SDLK_8:
                        densityDisplay = 0;
                        pressureDisplay = 0;
                        temperatureDisplay = 0;
                        gravPotentialDisplay = 0;
                        energyDisplay = 0;
                        hydrogenDisplay = 0;
                        heliumDisplay = !(heliumDisplay);
                        ZDisplay = 0;
                        fusionDisplay = 0;
                        degenerateDisplay = 0;
                        break;
                    case SDLK_9:
                        densityDisplay = 0;
                        pressureDisplay = 0;
                        temperatureDisplay = 0;
                        gravPotentialDisplay = 0;
                        energyDisplay = 0;
                        hydrogenDisplay = 0;
                        heliumDisplay = 0;
                        ZDisplay = !(ZDisplay);
                        fusionDisplay = 0;
                        degenerateDisplay = 0;
                        break;
                    case SDLK_f:
                        densityDisplay = 0;
                        pressureDisplay = 0;
                        temperatureDisplay = 0;
                        gravPotentialDisplay = 0;
                        energyDisplay = 0;
                        hydrogenDisplay = 0;
                        heliumDisplay = 0;
                        ZDisplay = 0;
                        fusionDisplay = !(fusionDisplay);
                        degenerateDisplay = 0;
                        break;
                    case SDLK_o:
                        densityDisplay = 0;
                        pressureDisplay = 0;
                        temperatureDisplay = 0;
                        gravPotentialDisplay = 0;
                        energyDisplay = 0;
                        hydrogenDisplay = 0;
                        heliumDisplay = 0;
                        ZDisplay = 0;
                        fusionDisplay = 0;
                        degenerateDisplay = !(degenerateDisplay);
                        break;
                    case SDLK_g:
                        gridDisplay = !(gridDisplay);
                        break;
                    case SDLK_v:
                        velocityDisplay = !(velocityDisplay);
                        break;
                    case SDLK_m:
                        mouseValueDisplay = !(mouseValueDisplay);
                }
            }
            if (display) {
                drawCells();
            }

            if (save) {
                saveSim();
            }
            // SDL_Delay(grid->getdt()*1000);  // setting some Delay
            SDL_Delay(16);
        }

        void saveSim() {
            for (auto &pair : grid->getIDMap()) {
                uint64_t ID = pair.first;
                FluidCell *cell = pair.second;
                long double mass = cell->getMass();
                long double pressure = cell->getPressure(false);
                long double density = cell->getDensity();
                long double temperature = cell->getTemp(false);
                long double gravPotential = cell->getGravPotential();
                long double e = cell->gete(false);
                long double fusion = cell->getFusionEnergy();
                output << "//" << ID << ":" << density << "," << cell->getX() << "," << cell->getY() << "," << cell->getZ() << "," << temperature << "," << pressure << "," << e << "," << gravPotential << "," << fusion;
            }
            output << "\nt:" << t;
            output << "\n";
        }

        void freeSim() {
            grid->freeGrid();
            free(grid);
            output.close();
        }

        double getT() {
            // time passed
            return t;
        }

        FluidGrid *grid = nullptr;
    private:
        // in meters per pixel
        double SCALE_H, SCALE_W;
        double width, height;
        bool display, save;
        double dt;
        SDL_Renderer *renderer = NULL;
        SDL_Window *window = NULL;
        SDL_Surface *screenSurface;
        std::ofstream output = std::ofstream();
        std::string filename = "";
        double t = 0;
        double maxMass = 255;
        double maxPressure = 1;
        double maxDensity = 1;
        double minDensity = 1e10;
        double maxTemperature = 1;
        double minTemperature = 0;
        double minGP = 0;
        double maxe = 0;
        double maxFusion = 0;
	    double maxPG = 0;
        uint pressureDisplay = 0;
        uint temperatureDisplay = 0;
        uint densityDisplay = 1;
        uint gravPotentialDisplay = 0;
        uint energyDisplay = 0;
        uint heliumDisplay = 0;
        uint hydrogenDisplay = 0;
        uint ZDisplay = 0;
        uint fusionDisplay = 0;
        uint degenerateDisplay = 0;
        uint gridDisplay = 0;
        uint velocityDisplay = 0;
        uint mouseValueDisplay = 0;
};

