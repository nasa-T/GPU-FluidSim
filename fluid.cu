#include <cmath>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>
//#include "fluid.h"
#include "fluid.cpp"

#include <map>
#include <string>
#include <functional>

#include <omp.h>
#include <algorithm>

#include <chrono>
#include <cassert>

__device__ double getFlux2(FluidCell *cell, double dist, double boundary, long double (*func)(FluidCell*), Direction dir, double totFlux, bool forceFarOff=true, bool farCell=false, void (*func2)(FluidCell*, double)=[](FluidCell *c, double m){c->newMass -= m;}) {
	if (!std::isfinite(dist)) return 0;
	double flux = 0;
	double leng = (dir == LEFT) || (dir == RIGHT) ? cell->getWidth() : cell->getHeight();
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

__global__ void advect_kernel(std::vector<FluidCell*> leaves, bool& minSizeSmall, double minSize, float maxFracTraveled, double& thisMaxDensity, double& maxDensity, double& maxV, float dt) {
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

		if (cell->getDensity() > 1e-2*maxDensity && fmax(abs(vC.getVx()),abs(vC.getVy())) > maxV) {
			maxV = fmax(abs(vC.getVx()),abs(vC.getVy()));
		}
		float fracTraveled = fmax(abs(vC.getVx()),abs(vC.getVy()))*dt/std::sqrt(cell->getSize());
		if (fracTraveled > maxFracTraveled) {
			if (fracTraveled > 1) {
				// std::cout << "maxFrac: " << fracTraveled << " " << (cell->getMass()-cellR->getMass())/(2*cell->getWidth()) << std::endl;
			}
			maxFracTraveled = fracTraveled;
		}

		// avgFracTraveled += std::isfinite(fracTraveled) ? fracTraveled : 0;

		if (cell->getHeight() == minSize || cell->getWidth() == minSize) minSizeSmall = false;

		double massFluxR = 0;
		double massFluxL = 0;
		double massFluxT = 0;
		double massFluxB = 0;
		double EFluxR = 0;
		double EFluxL = 0;
		double EFluxT = 0;
		double EFluxB = 0;
		double vxFluxR = 0;
		double vyFluxR = 0;
		double vxFluxL = 0;
		double vyFluxL = 0;
		double vxFluxT = 0;
		double vyFluxT = 0;
		double vxFluxB = 0;
		double vyFluxB = 0;
		double XFluxR = 0;
		double XFluxL = 0;
		double XFluxT = 0;
		double XFluxB = 0;
		double YFluxR = 0;
		double YFluxL = 0;
		double YFluxT = 0;
		double YFluxB = 0;
		double ZFluxR = 0;
		double ZFluxL = 0;
		double ZFluxT = 0;
		double ZFluxB = 0;
		double unFluxR = 0;
		double unFluxL = 0;
		double unFluxT = 0;
		double unFluxB = 0;
		double vxR;
		double vxL;
		double vyT;
		double vyB;
		VelocityVector vR, vL, vT, vB;
		double mass = cell->getMass();
		double E = cell->getE(false);

		if (cell->getDensity() > thisMaxDensity) thisMaxDensity = cell->getDensity();
		vxR = (cell->getRight()->getVelocity().getVx() + vC.getVx())/2;
		vxL = (cell->getLeft()->getVelocity().getVx() + vC.getVx())/2;
		vyT = (cell->getTop()->getVelocity().getVy() + vC.getVy())/2;
		vyB = (cell->getBottom()->getVelocity().getVy() + vC.getVy())/2;
		// vR = cellR->getVelocity();
		// vL = cellL->getVelocity();
		// vT = cellT->getVelocity();
		// vB = cellB->getVelocity();
		Direction dir;
		FluidCell *nextCell;

		double dist, face;

		// not a boundary
		if (cellR != cell) {
			if (cellR->hasChildren()) {
				FluidCell *children[2] = {cellR->nw, cellR->sw};
				for (int i = 0; i < 2; i++) {
					vxR = (children[i]->getVelocity().getVx() + vC.getVx())/2;
					if (vxR < 0) {
						dir = RIGHT;
						nextCell = children[i];
						dist = fabs(vxR*dt);
					} else {
						dir = LEFT;
						nextCell = cell;
						dist = fmin(fabs(vxR*dt),cell->getWidth());
					}
					face = children[i]->getHeight();
					
					massFluxR += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass();}, dir, 0, false, false);
					EFluxR += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getE(false)*c->getSize();}, dir, 0, false, false, [](FluidCell *c, double m){c->newE -= m/c->getSize();});
					// vxFluxR += getFlux(nextCell, fabs(vxR*dt), children[i]->getHeight(), MOMENTUM, dir, 0);
					vxFluxR += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass()*c->getVelocity().getVx();}, dir, 0);
					vyFluxR += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass()*c->getVelocity().getVy();}, dir, 0);
					XFluxR += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getX()*c->getMass();}, dir, 0);
					YFluxR += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getY()*c->getMass();}, dir, 0);
					ZFluxR += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getZ()*c->getMass();}, dir, 0);
					unFluxR += getFlux2(nextCell, dist, face, [](FluidCell *c) {return (long double) 1;}, dir, 0);
				}
				
			} else {
				if (vxR < 0) {
					dir = RIGHT;
					nextCell = cellR;
					dist = fabs(vxR*dt);
				} else {
					dir = LEFT;
					nextCell = cell;
					dist = fmin(fabs(vxR*dt),cell->getWidth());
				}
				face = cell->getHeight();

				massFluxR = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass();}, dir, 0, false, false);
				EFluxR = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getE(false)*c->getSize();}, dir, 0, false, false, [](FluidCell *c, double m){c->newE -= m/c->getSize();});
				// vxFluxR = getFlux(nextCell, fabs(vxR*dt), cell->getHeight(), MOMENTUM, dir, 0);
				vxFluxR = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass()*c->getVelocity().getVx();}, dir, 0);
				vyFluxR = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass()*c->getVelocity().getVy();}, dir, 0);
				XFluxR = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getX()*c->getMass();}, dir, 0);
				YFluxR = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getY()*c->getMass();}, dir, 0);
				ZFluxR = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getZ()*c->getMass();}, dir, 0);
				unFluxR = getFlux2(nextCell, dist, face, [](FluidCell *c) {return (long double) 1;}, dir, 0);
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
					if (vxL < 0) {
						dir = RIGHT;
						nextCell = cell;
						dist = fmin(fabs(vxL*dt),cell->getWidth());
					} else {
						dir = LEFT;
						nextCell = children[i];
						dist = fabs(vxL*dt);
					}
					face = children[i]->getHeight();

					massFluxL += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass();}, dir, 0, false, false);
					EFluxL += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getE(false)*c->getSize();}, dir, 0, false, false, [](FluidCell *c, double m){c->newE -= m/c->getSize();});
					// vxFluxL += getFlux(nextCell, fabs(vxL*dt), children[i]->getHeight(), MOMENTUM, dir, 0);
					vxFluxL += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass()*c->getVelocity().getVx();}, dir, 0);
					vyFluxL += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass()*c->getVelocity().getVy();}, dir, 0);
					XFluxL += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getX()*c->getMass();}, dir, 0);
					YFluxL += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getY()*c->getMass();}, dir, 0);
					ZFluxL += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getZ()*c->getMass();}, dir, 0);
					unFluxL += getFlux2(nextCell, dist, face, [](FluidCell *c) {return (long double) 1;}, dir, 0);
				}
			} else {
				if (vxL < 0) {
					dir = RIGHT;
					nextCell = cell;
					dist = fmin(fabs(vxL*dt),cell->getWidth());
				} else {
					dir = LEFT;
					nextCell = cellL;
					dist = fabs(vxL*dt);
				}
				face = cell->getHeight();

				massFluxL = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass();}, dir, 0, false, false);
				EFluxL = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getE(false)*c->getSize();}, dir, 0, false, false, [](FluidCell *c, double m){c->newE -= m/c->getSize();});
				// vxFluxL = getFlux(nextCell, fabs(vxL*dt), cell->getHeight(), MOMENTUM, dir, 0);
				vxFluxL = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass()*c->getVelocity().getVx();}, dir, 0);
				vyFluxL = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass()*c->getVelocity().getVy();}, dir, 0);
				XFluxL = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getX()*c->getMass();}, dir, 0);
				YFluxL = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getY()*c->getMass();}, dir, 0);
				ZFluxL = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getZ()*c->getMass();}, dir, 0);
				unFluxL = getFlux2(nextCell, dist, face, [](FluidCell *c) {return (long double) 1;}, dir, 0);
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
					if (vyT < 0) {
						dir = UP;
						nextCell = children[i];
						dist = fabs(vyT*dt);
					} else {
						dir = DOWN;
						nextCell = cell;
						dist = fmin(fabs(vyT*dt),cell->getHeight());
					}

					face = children[i]->getWidth();
					massFluxT += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass();}, dir, 0, false, false);
					EFluxT += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getE(false)*c->getSize();}, dir, 0, false, false, [](FluidCell *c, double m){c->newE -= m/c->getSize();});
					vxFluxT += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass()*c->getVelocity().getVx();}, dir, 0);
					vyFluxT += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass()*c->getVelocity().getVy();}, dir, 0);
					XFluxT += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getX()*c->getMass();}, dir, 0);
					YFluxT += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getY()*c->getMass();}, dir, 0);
					ZFluxT += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getZ()*c->getMass();}, dir, 0);
					unFluxT += getFlux2(nextCell, dist, face, [](FluidCell *c) {return (long double) 1;}, dir, 0);
				}
				
			} else {
				if (vyT < 0) {
					dir = UP;
					nextCell = cellT;
					dist = fabs(vyT*dt);
				} else {
					dir = DOWN;
					nextCell = cell;
					dist = fmin(fabs(vyT*dt),cell->getHeight());
				}

				face = cell->getWidth();
				massFluxT = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass();}, dir, 0, false, false);
				EFluxT = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getE(false)*c->getSize();}, dir, 0, false, false, [](FluidCell *c, double m){c->newE -= m/c->getSize();});
				vxFluxT = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass()*c->getVelocity().getVx();}, dir, 0);
				vyFluxT = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass()*c->getVelocity().getVy();}, dir, 0);
				XFluxT = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getX()*c->getMass();}, dir, 0);
				YFluxT = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getY()*c->getMass();}, dir, 0);
				ZFluxT = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getZ()*c->getMass();}, dir, 0);
				unFluxT = getFlux2(nextCell, dist, face, [](FluidCell *c) {return (long double) 1;}, dir, 0);
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
					if (vyB < 0) {
						dir = UP;
						nextCell = cell;
						dist = fmin(fabs(vyB*dt),cell->getHeight());
					} else {
						dir = DOWN;
						nextCell = children[i];
						dist = fabs(vyB*dt);
					}

					face = children[i]->getWidth();
					massFluxB += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass();}, dir, 0, false, false);
					EFluxB += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getE(false)*c->getSize();}, dir, 0, false, false, [](FluidCell *c, double m){c->newE -= m/c->getSize();});
					vxFluxB += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass()*c->getVelocity().getVx();}, dir, 0);
					vyFluxB += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass()*c->getVelocity().getVy();}, dir, 0);
					XFluxB += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getX()*c->getMass();}, dir, 0);
					YFluxB += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getY()*c->getMass();}, dir, 0);
					ZFluxB += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getZ()*c->getMass();}, dir, 0);
					unFluxB += getFlux2(nextCell, dist, face, [](FluidCell *c) {return (long double) 1;}, dir, 0);
				}
				
			} else {
				if (vyB < 0) {
					dir = UP;
					nextCell = cell;
					dist = fmin(fabs(vyB*dt),cell->getHeight());
				} else {
					dir = DOWN;
					nextCell = cellB;
					dist = fabs(vyB*dt);
				}

				face = cell->getWidth();
				massFluxB = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass();}, dir, 0, false, false);
				EFluxB = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getE(false)*c->getSize();}, dir, 0, false, false, [](FluidCell *c, double m){c->newE -= m/c->getSize();});
				vxFluxB = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass()*c->getVelocity().getVx();}, dir, 0);
				vyFluxB = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getMass()*c->getVelocity().getVy();}, dir, 0);
				XFluxB = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getX()*c->getMass();}, dir, 0);
				YFluxB = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getY()*c->getMass();}, dir, 0);
				ZFluxB = getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->getZ()*c->getMass();}, dir, 0);
				unFluxB = getFlux2(nextCell, dist, face, [](FluidCell *c) {return (long double) 1;}, dir, 0);
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
			massFluxR = fmax((double) 0, massFluxL);
			EFluxR = fmax((double) 0, EFluxL);
			vxFluxR = fmax((double) 0, vxFluxL);
			vyFluxR = fmax((double) 0, vyFluxL);
			XFluxR = fmax((double) 0, XFluxL);
			YFluxR = fmax((double) 0, YFluxL);
			ZFluxR = fmax((double) 0, ZFluxL);
		}

		if (cell == cellL) {
			massFluxL = fmin((double) 0, massFluxR);
			EFluxL = fmin((double) 0, EFluxR);
			vxFluxL = fmin((double) 0, vxFluxR);
			vyFluxL = fmin((double) 0, vyFluxR);
			XFluxL = fmin((double) 0, XFluxR);
			YFluxL = fmin((double) 0, YFluxR);
			ZFluxL = fmin((double) 0, ZFluxR);
		}

		if (cell == cellT) {
			massFluxT = fmax((double) 0, massFluxB);
			EFluxT = fmax((double) 0, EFluxB);
			vxFluxT = fmax((double) 0, vxFluxB);
			vyFluxT = fmax((double) 0, vyFluxB);
			XFluxT = fmax((double) 0, XFluxB);
			YFluxT = fmax((double) 0, YFluxB);
			ZFluxT = fmax((double) 0, ZFluxB);
		}

		if (cell == cellB) {
			massFluxB = fmin((double) 0, massFluxT);
			EFluxB = fmin((double) 0, EFluxT);
			vxFluxB = fmin((double) 0, vxFluxT);
			vyFluxB = fmin((double) 0, vyFluxT);
			XFluxB = fmin((double) 0, XFluxT);
			YFluxB = fmin((double) 0, YFluxT);
			ZFluxB = fmin((double) 0, ZFluxT);
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

		double massFluxes[4] = {massFluxR, -massFluxL, massFluxT, -massFluxB};
		double EFluxes[4] = {EFluxR, -EFluxL, EFluxT, -EFluxB};

		double massFluxOut = 0;
		double EFluxOut = 0;
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

		cell->newEx = fmax((double) 0.0f, (E*cell->getSize() + 0.5*(-EFluxR + EFluxL))/cell->getSize());
		cell->newEy = fmax((double) 0.0f, (E*cell->getSize() + 0.5*(-EFluxT + EFluxB))/cell->getSize());

		double newVx = (mass*cell->getVelocity().getVx() + -vxFluxR + vxFluxL + -vxFluxT + vxFluxB);
		double newVxx = (mass*cell->getVelocity().getVx() + 0.5*(-vxFluxR + vxFluxL))/cell->newMassx;
		double newVxy = (mass*cell->getVelocity().getVx() + 0.5*(-vxFluxT + vxFluxB))/cell->newMassy;
		double newVy = (mass*cell->getVelocity().getVy() + -vyFluxR + vyFluxL + -vyFluxT + vyFluxB);
		double newVyx = (mass*cell->getVelocity().getVy() + 0.5*(-vyFluxR + vyFluxL))/cell->newMassx;
		double newVyy = (mass*cell->getVelocity().getVy() + 0.5*(-vyFluxT + vyFluxB))/cell->newMassy;

		cell->newVelocityx = VelocityVector(newVxx, newVyx);
		cell->newVelocityy = VelocityVector(newVxy, newVyy);
		cell->newVelocity = VelocityVector(newVx, newVy);

		cell->newXx = fmin(1.0f,fmax(0.0f,(float)((mass*X + 0.5*(-XFluxR + XFluxL))/cell->newMassx)) );
		cell->newXy = fmin(1.0f,fmax(0.0f,(float)((mass*X + 0.5*(-XFluxT + XFluxB))/cell->newMassy)) );

		cell->newYx = fmin(1.0f,fmax(0.0f,(float)((mass*Y + 0.5*(-YFluxR + YFluxL))/cell->newMassx)) );
		cell->newYy = fmin(1.0f,fmax(0.0f,(float)((mass*Y + 0.5*(-YFluxT + YFluxB))/cell->newMassy)) );

		cell->newZx = fmin(1.0f,fmax(0.0f,(float)((mass*Z + 0.5*(-ZFluxR + ZFluxL))/cell->newMassx)) );
		cell->newZy = fmin(1.0f,fmax(0.0f,(float)((mass*Z + 0.5*(-ZFluxT + ZFluxB))/cell->newMassy)) );

	}
}

__global__ void advect_kernel2(std::vector<FluidCell*> leaves) {
	int workIdx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (workIdx < leaves.size()) {
		FluidCell *cell = leaves[workIdx];
		double mass = cell->getMass();
		double E = cell->getE(false);

		double massFluxOut = abs(cell->newMass);
		double EFluxOut = abs(cell->newE*cell->getSize());
		
		cell->massODs = fmin((double) 1.0f, mass/massFluxOut);
		cell->EODs = fmin((double) 1.0f, E*cell->getSize()/EFluxOut);

		cell->newMass = 0;
		cell->newE = 0;
	}
}

__global__ void advect_kernel3(std::vector<FluidCell*> leaves, bool& minSizeSmall, double minSize, double& thisMaxDensity, double& maxDensity, float dt) {
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

		double massFluxR = 0;
		double massFluxL = 0;
		double massFluxT = 0;
		double massFluxB = 0;
		double EFluxR = 0;
		double EFluxL = 0;
		double EFluxT = 0;
		double EFluxB = 0;
		double vxFluxR = 0;
		double vyFluxR = 0;
		double vxFluxL = 0;
		double vyFluxL = 0;
		double vxFluxT = 0;
		double vyFluxT = 0;
		double vxFluxB = 0;
		double vyFluxB = 0;
		double XFluxR = 0;
		double XFluxL = 0;
		double XFluxT = 0;
		double XFluxB = 0;
		double YFluxR = 0;
		double YFluxL = 0;
		double YFluxT = 0;
		double YFluxB = 0;
		double ZFluxR = 0;
		double ZFluxL = 0;
		double ZFluxT = 0;
		double ZFluxB = 0;
		double vxR;
		double vxL;
		double vyT;
		double vyB;
		double mass = cell->getMass();
		double E = cell->getE(false);
		double dist, face;

		if (cell->getDensity() > thisMaxDensity) thisMaxDensity = cell->getDensity();
		vxR = (cell->getRight()->getVelocity().getVx() + vC.getVx())/2;
		vxL = (cell->getLeft()->getVelocity().getVx() + vC.getVx())/2;
		vyT = (cell->getTop()->getVelocity().getVy() + vC.getVy())/2;
		vyB = (cell->getBottom()->getVelocity().getVy() + vC.getVy())/2;

		Direction dir;
		FluidCell *nextCell;

		// not a boundary
		if (cellR != cell) {
			if (cellR->hasChildren()) {
				FluidCell *children[2] = {cellR->nw, cellR->sw};
				for (int i = 0; i < 2; i++) {
					vxR = (children[i]->getVelocity().getVx() + vC.getVx())/2;
					if (vxR < 0) {
						dir = RIGHT;
						nextCell = children[i];
						dist = fabs(vxR*dt);
					} else {
						dir = LEFT;
						nextCell = cell;
						dist = fmin(fabs(vxR*dt),cell->getWidth());
					}

					face = children[i]->getHeight();
					massFluxR += getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newMassy*c->massODs;}, dir, 0, false, false, [](FluidCell *c, double m){c->newMass -= m;});
					EFluxR += getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newEy*c->getSize()*c->EODs;}, dir, 0, false, false, [](FluidCell *c, double m){c->newE -= m/c->getSize();});
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
					dist = fabs(vxR*dt);
				} else {
					dir = LEFT;
					nextCell = cell;
					dist = fmin(fabs(vxR*dt),cell->getWidth());
				}

				face = cell->getHeight();
				massFluxR = getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newMassy*c->massODs;}, dir, 0, false, false, [](FluidCell *c, double m){c->newMass -= m;});
				EFluxR = getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newEy*c->getSize()*c->EODs;}, dir, 0, false, false, [](FluidCell *c, double m){c->newE -= m/c->getSize();});
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
					if (vxL < 0) {
						dir = RIGHT;
						nextCell = cell;
						dist = fmin(fabs(vxL*dt),cell->getWidth());
					} else {
						dir = LEFT;
						nextCell = children[i];
						dist = fabs(vxL*dt);
					}

					face = children[i]->getHeight();
					massFluxL += getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newMassy*c->massODs;}, dir, 0, false, false, [](FluidCell *c, double m){c->newMass -= m;});
					EFluxL += getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newEy*c->getSize()*c->EODs;}, dir, 0, false, false, [](FluidCell *c, double m){c->newE -= m/c->getSize();});
					vxFluxL += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->newMassy*c->newVelocityy.getVx()*c->massODs;}, dir, 0);
					vyFluxL += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->newMassy*c->newVelocityy.getVy()*c->massODs;}, dir, 0);
					XFluxL += getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newXy*c->newMassy*c->massODs;}, dir, 0);
					YFluxL += getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newYy*c->newMassy*c->massODs;}, dir, 0);
					ZFluxL += getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newZy*c->newMassy*c->massODs;}, dir, 0);
				}
			} else {
				if (vxL < 0) {
					dir = RIGHT;
					nextCell = cell;
					dist = fmin(fabs(vxL*dt),cell->getWidth());
				} else {
					dir = LEFT;
					nextCell = cellL;
					dist = fabs(vxL*dt);
				}

				face = cell->getHeight();
				massFluxL = getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newMassy*c->massODs;}, dir, 0, false, false, [](FluidCell *c, double m){c->newMass -= m;});
				EFluxL = getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newEy*c->getSize()*c->EODs;}, dir, 0, false, false, [](FluidCell *c, double m){c->newE -= m/c->getSize();});
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
					if (vyT < 0) {
						dir = UP;
						nextCell = children[i];
						dist = fabs(vyT*dt);
					} else {
						dir = DOWN;
						nextCell = cell;
						dist = fmin(fabs(vyT*dt),cell->getHeight());
					}

					face = children[i]->getWidth();
					massFluxT += getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newMassx*c->massODs;}, dir, 0, false, false, [](FluidCell *c, double m){c->newMass -= m;});
					EFluxT += getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newEx*c->getSize()*c->EODs;}, dir, 0, false, false, [](FluidCell *c, double m){c->newE -= m/c->getSize();});
					vxFluxT += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->newMassx*c->newVelocityx.getVx()*c->massODs;}, dir, 0);
					vyFluxT += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->newMassx*c->newVelocityx.getVy()*c->massODs;}, dir, 0);
					XFluxT += getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newXx*c->newMassx*c->massODs;}, dir, 0);
					YFluxT += getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newYx*c->newMassx*c->massODs;}, dir, 0);
					ZFluxT += getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newZx*c->newMassx*c->massODs;}, dir, 0);
				}
				
			} else {
				if (vyT < 0) {
					dir = UP;
					nextCell = cellT;
					dist = fabs(vyT*dt);
				} else {
					dir = DOWN;
					nextCell = cell;
					dist = fmin(fabs(vyT*dt),cell->getHeight());
				}

				face = cell->getWidth();
				massFluxT = getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newMassx*c->massODs;}, dir, 0, false, false, [](FluidCell *c, double m){c->newMass -= m;});
				EFluxT = getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newEx*c->getSize()*c->EODs;}, dir, 0, false, false, [](FluidCell *c, double m){c->newE -= m/c->getSize();});
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
					if (vyB < 0) {
						dir = UP;
						nextCell = cell;
						dist = fmin(fabs(vyB*dt),cell->getHeight());
					} else {
						dir = DOWN;
						nextCell = children[i];
						dist = fabs(vyB*dt);
					}
					face = children[i]->getWidth();
					massFluxB += getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newMassx*c->massODs;}, dir, 0, false, false, [](FluidCell *c, double m){c->newMass -= m;});
					EFluxB += getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newEx*c->getSize()*c->EODs;}, dir, 0, false, false, [](FluidCell *c, double m){c->newE -= m/c->getSize();});

					vxFluxB += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->newMassx*c->newVelocityx.getVx()*c->massODs;}, dir, 0);
					vyFluxB += getFlux2(nextCell, dist, face, [](FluidCell *c) {return c->newMassx*c->newVelocityx.getVy()*c->massODs;}, dir, 0);
					XFluxB += getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newXx*c->newMassx*c->massODs;}, dir, 0);
					YFluxB += getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newYx*c->newMassx*c->massODs;}, dir, 0);
					ZFluxB += getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newZx*c->newMassx*c->massODs;}, dir, 0);
				}
				
			} else {
				if (vyB < 0) {
					dir = UP;
					nextCell = cell;
					dist = fmin(fabs(vyB*dt),cell->getHeight());
				} else {
					dir = DOWN;
					nextCell = cellB;
					dist = fabs(vyB*dt);
				}
				face = cell->getWidth();
				massFluxB = getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newMassx*c->massODs;}, dir, 0, false, false, [](FluidCell *c, double m){c->newMass -= m;});
				EFluxB = getFlux2(nextCell, dist, face, [](FluidCell *c){return c->newEx*c->getSize()*c->EODs;}, dir, 0, false, false, [](FluidCell *c, double m){c->newE -= m/c->getSize();});

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
			massFluxR = fmax((double) 0, massFluxL);
			EFluxR = fmax((double) 0, EFluxL);
			vxFluxR = fmax((double) 0, vxFluxL);
			vyFluxR = fmax((double) 0, vyFluxL);
			XFluxR = fmax((double) 0, XFluxL);
			YFluxR = fmax((double) 0, YFluxL);
			ZFluxR = fmax((double) 0, ZFluxL);
		}

		if (cell == cellL) {
			massFluxL = fmin((double) 0, massFluxR);
			EFluxL = fmin((double) 0, EFluxR);
			vxFluxL = fmin((double) 0, vxFluxR);
			vyFluxL = fmin((double) 0, vyFluxR);
			XFluxL = fmin((double) 0, XFluxR);
			YFluxL = fmin((double) 0, YFluxR);
			ZFluxL = fmin((double) 0, ZFluxR);
		}

		if (cell == cellT) {
			massFluxT = fmax((double) 0, massFluxB);
			EFluxT = fmax((double) 0, EFluxB);
			vxFluxT = fmax((double) 0, vxFluxB);
			vyFluxT = fmax((double) 0, vyFluxB);
			XFluxT = fmax((double) 0, XFluxB);
			YFluxT = fmax((double) 0, YFluxB);
			ZFluxT = fmax((double) 0, ZFluxB);
		}

		if (cell == cellB) {
			massFluxB = fmin((double) 0, massFluxT);
			EFluxB = fmin((double) 0, EFluxT);
			vxFluxB = fmin((double) 0, vxFluxT);
			vyFluxB = fmin((double) 0, vyFluxT);
			XFluxB = fmin((double) 0, XFluxT);
			YFluxB = fmin((double) 0, YFluxT);
			ZFluxB = fmin((double) 0, ZFluxT);
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

		cell->newX = fmax((double) 0.0f,((mass*X - XFluxR + XFluxL - XFluxT + XFluxB)));
		cell->newY = fmax((double) 0.0f,((mass*Y - YFluxR + YFluxL - YFluxT + YFluxB)));
		cell->newZ = fmax((double) 0.0f,((mass*Z - ZFluxR + ZFluxL - ZFluxT + ZFluxB)));
	}
}

__global__ void advect_kernel4(std::vector<FluidCell*> leaves, double& totMass) {
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
			totMass += fmax((double) 0.0f, cell->newMass);
			cell->setVelocity(cell->newVelocity/cell->newMass);
			double factor = cell->newMass/(cell->newX+cell->newY+cell->newZ);

			cell->setX(fmax(0.0f,fmin(1.0f,(float) (factor*cell->newX/cell->newMass))));
			cell->setY(fmax(0.0f,fmin(1.0f,(float) (factor*cell->newY/cell->newMass))));
			cell->setZ(fmax(0.0f,fmin(1.0f,(float) (factor*cell->newZ/cell->newMass))));
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

void launch_advect_kernel(int blocks, int threads, std::vector<FluidCell*> leaves, int step, bool& minSizeSmall, double minSize, float maxFracTraveled, double& thisMaxDensity, double& maxDensity, double& totMass, double& maxV, float dt) {
	if (step == 1) {
		advect_kernel<<<blocks, threads>>>(leaves, minSizeSmall, minSize, maxFracTraveled, thisMaxDensity, maxDensity, maxV, dt);
	} else if (step == 2) {
		advect_kernel2<<<blocks, threads>>>(leaves);
	} else if (step == 3) {
		advect_kernel3<<<blocks, threads>>>(leaves, minSizeSmall, minSize, thisMaxDensity, maxDensity, dt);
	} else if (step == 4) {
		advect_kernel4<<<blocks, threads>>>(leaves, totMass);
	}
	cudaDeviceSynchronize();
}

