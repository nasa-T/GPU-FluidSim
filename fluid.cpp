#include <cmath>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>
//#include "fluid.h"

#include <map>
#include <string>
//#include <functional>

#include <omp.h>
#include <algorithm>

#include <chrono>
#include <cassert>

#include <cuda_runtime.h>

class VelocityVector {
    public:
        VelocityVector(double vx_ = 0, double vy_ = 0, double ax_ = 0, double ay_ = 0): vx(vx_), vy(vy_), ax(ax_), ay(ay_) {}
        VelocityVector operator+(VelocityVector& other) {
            VelocityVector added(this->vx + other.vx, this->vy + other.vy, other.ax, other.ay);
            return added;
        }
        VelocityVector operator/(double d) {
            VelocityVector divided(this->vx/d, this->vy/d, ax, ay);
            return divided;
        }
        VelocityVector operator-(VelocityVector& other) {
            VelocityVector subtracted(this->vx - other.vx, this->vy - other.vy, other.ax, other.ay);
            return subtracted;
        }
        VelocityVector operator*(double d) {
            VelocityVector divided(this->vx*d, this->vy*d, ax, ay);
            return divided;
        }
        double getVx() {
            return vx;
        }
        double getVy() {
            return vy;
        }
        double getAx() {
            return ax;
        }
        double getAy() {
            return ay;
        }
        void setVx(double v) {
            vx = v;
        }
        void setVy(double v) {
            vy = v;
        }
        void accelerate() {
            vx += ax * consts::dt;
            vy += ay * consts::dt;
        }
        double getMag() {
            return std::sqrt(getVx()*getVx()+getVy()*getVy());
            // return pow(vx*vx+vy*vy,1/2);
        }
        // VelocityVector operator+(VelocityVector &vec) {
        //     return VelocityVector(vx+vec.getVx(), vy+vec.getVy());
        // }
        // std::ostream& operator<<(std::ostream &s, VelocityVector &vec) {
        //     return s << "vx: " << vec.getVx() << " vy: " << vec.getVy();
        // }
    private:
        double vx, vy, ax, ay;
};

struct Neighbors {
    FluidCell *top, *bottom, *left, *right = nullptr;
};

class FluidCell {
    public:
        FluidCell(FluidCell *parent, uint32_t depth, uint32_t idx, long double mass, double width, double height, double temp, float X=0.8, float Y=0.15, float Z=0.05): parent(parent), depth(depth), index(idx), mass(mass), width(width), height(height), temperature(temp), nw(nullptr), ne(nullptr), sw(nullptr), se(nullptr) {
            // size is the physical area
            size = width * height;
            density = mass/size;
            setX(X);
            setY(Y);
            setZ(Z);
            long double nonMet = (hydrogen+helium);
            long double cv = consts::r/(consts::HMol*hydrogen/nonMet+consts::HeMol*helium/nonMet)/(1.4-1);
            e = cv*temperature*density;
            // std::cout << e << std::endl;
			double kap = opacity(getDensity(),getTemp(false),getX());
			double tau = opticalDepth(kap, getDensity(), sqrt(getSize()));
			double rad_pressure = 1/3*consts::sig_sb*pow(temperature,4)/consts::c*(1 - exp(-tau));
            pressure = (1.4-1)*e + rad_pressure;
            velocity = VelocityVector();
            neighbors = Neighbors();
            double gravPotential = 0;
            nw = nullptr; ne = nullptr; sw = nullptr; se = nullptr;
        }

        long double getCv() {
            double nonMet = (hydrogen+helium);
            return consts::r/(consts::HMol*hydrogen/nonMet+consts::HeMol*helium/nonMet)/(1.4-1);
        }

        long double getMass() {
	    absorbChildProps();
            return mass;
        }

        long double getSize() {
            return size;
        }

        long double getWidth() {
            return width;
        }

        long double getHeight() {
            return height;
        }

        long double getDensity() {
	    absorbChildProps();
            return mass/size;
        }

        long double getTemp(bool recalculate=true) {
            if (recalculate) {
                // double cv = consts::r/consts::HMol/(1.4-1);
                if (mass == 0) {
                    // std::cout << mass << std::endl;
                    temperature = 0;
                } else {
                    temperature = gete(false)/(getDensity()*getCv());
                }
            }
            return temperature;
        }

        void setTemp(long double t) {
            temperature = t;
        }

        double getPressure(bool recalculate=true) {
            // double oldPressure = pressure;
            if (recalculate) {
                // pressure = getDensity()/consts::HMol * consts::r * getTemp();
				double kap = opacity(getDensity(),getTemp(false),getX());
				double tau = opticalDepth(kap, getDensity(), sqrt(getSize()));
				double rad_pressure = 1/3*consts::sig_sb*pow(temperature,4)/consts::c*(1 - exp(-tau));
                pressure = (1.4 - 1)*gete(false) + rad_pressure;
            }
            
            double degen_pressure = 1/8*std::pow(3/consts::PI,1/3)*consts::h*consts::c*std::pow(getDensity()/consts::mH, 4/3);
            pressure = std::max(degen_pressure, pressure);
            if (pressure == degen_pressure) degenerate = 1;
            else degenerate = 0;			
        
            return pressure;
        }

        void setPressure(double p) {
            pressure = p;
        }

        void setMass(long double m) {
            if (m > 1e-8*getSize() && std::isfinite(m)) {
                mass = m;
            } else {
                mass = 1e-8*getSize();
                // e = 0;
                // E = 0;
                // pressure = 0;
                // temperature = 0;
            }
        }

        void addMass(double m) {
            setMass(mass + m);
        }

        float getX() {
            // hydrogen = hydrogen/(hydrogen+helium+metals);
            return hydrogen;
        }

        void setX(float h) {
            if (h < 0) hydrogen = 0;
            else if (h > 1)	hydrogen = 1;
            else hydrogen = h;
            // if (round(mass) > 0) {

            // } else {
            //     // std::cout << mass << " " << h << std::endl;
            //     hydrogen = 0;
            //     // hydrogen = h;
            // }

        }

        float getY() {
            // helium = helium/(hydrogen+helium+metals);
            return helium;
        }

        void setY(float he) {
            if (he < 0) helium = 0;
            else if (he > 1) helium = 1;
            else helium = he;
            // if (round(mass) > 0) {
                
            // } else {
            //     helium = 0;
            //     // helium = he;
            // }
        }

        float getZ() {
            // metals = 1 - hydrogen - helium;
            // metals = metals/(hydrogen+helium+metals);
            return metals;
        }

        void setZ(float Z) {
            if (Z < 0) metals = 0;
            else if (Z > 1) metals = 1;
            else metals = Z;
            
        }

        void HtoHe(float dX) {
            if (getX() - dX < 0) {
                setX(0);
                setY(1-getZ());
            } else {
                setX(getX() - dX);
                setY(getY() + dX);
            }
        }

        void HetoZ(float dY) {
            if (getY() - dY < 0) {
                setY(0);
                setZ(1-getX());
            } else {
                setY(getY() - dY);
                setZ(getZ() + dY);
            }
        }

        long double gete(bool recalculate=true) {
            if (recalculate) {
                // if (round(getDensity()*1000) == 0) e = 0;
                // else 
                e = getCv()*getTemp(false)*getDensity();
                // e = E - 1/2*getDensity()*velocity.getMag()*velocity.getMag();
            }
            return e;
        }
	
        void sete(long double energy) {
            if (mass > 0 && energy > 0) {
                e = energy;
            } else {
                e = 0;
                // mass = 0;
                // E = 0;
            }
        }
        
        long double getE(bool recalculate=false) {
            // if (recalculate) {
            //     E = e + 1/2*getDensity()*velocity.getMag()*velocity.getMag();
            // }
            return E;
        }

        void setE(long double energy) {
            if (mass > 0 && energy > 0) {
                E = energy;
            } else {
                E = 0;
                // mass = 0;
                // e = 0;
            }
        }

	void pressGrad(long double pg) {
	    PG = pg;
	}
	
        uint getDegenerate() {
            return degenerate;
        }

        double getFusionEnergy() {
            return fusionE;
        }

        void setFusionEnergy(double fE) {
            fusionE = fE;
        }

        double getGravPotential() {
            return gravPotential;
        }

        void setGravPotential(double u) {
            gravPotential = u;
        }

        FluidCell *getRight() {
            return neighbors.right;
        }
        FluidCell *getLeft() {
            return neighbors.left;
        }
        FluidCell *getTop() {
            return neighbors.top;
        }
        FluidCell *getBottom() {
            return neighbors.bottom;
        }

        void setRight(FluidCell *neighbor) {
            neighbors.right = neighbor;
        }
        void setLeft(FluidCell *neighbor) {
            neighbors.left = neighbor;
        }
        void setTop(FluidCell *neighbor) {
            neighbors.top = neighbor;
        }
        void setBottom(FluidCell *neighbor) {
            neighbors.bottom = neighbor;
        }

        void transferMass(FluidCell *other, double m) {
            if (other != NULL) { // halt at the boundaries; NOTHING GETS OUT
                if (m > 0) {
                    if (this->getMass() < m) {
                        m = this->getMass();
                    }
                    double otherMass = other->getMass();
                    other->setMass(otherMass + m);
                    this->setMass(mass - m);
                } else if (m < 0) {
                    m = -m;
                    if (other->getMass() < m) {
                        m = other->getMass();
                    }
                    double otherMass = other->getMass();
                    other->setMass(otherMass - m);
                    this->setMass(mass + m);
                }
            }
        }

        void setVelocity(double vx, double vy) {
            velocity.setVx(vx);
            velocity.setVy(vy);
        }

        void setVelocity(VelocityVector v) {
			if (v.getMag() == 0.0) {
				//std::cout << "zero vel dens: " << getDensity() << std::endl;
			}
            velocity = v;
        }

        VelocityVector getVelocity() {
            // return VelocityVector(0,1);
            return velocity;
        }

        void refine() {
            refinedThisStep = true;
            if (hasChildren()) return;

            nw = new FluidCell(this, depth+1, (index << 2) | 0, mass/4, width/2, height/2, temperature, hydrogen, helium, metals);
            nw->setGravPotential(gravPotential);
            ne = new FluidCell(this, depth+1, (index << 2) | 1, mass/4, width/2, height/2, temperature, hydrogen, helium, metals);
            ne->setGravPotential(gravPotential);
            sw = new FluidCell(this, depth+1, (index << 2) | 2, mass/4, width/2, height/2, temperature, hydrogen, helium, metals);
            sw->setGravPotential(gravPotential);
            se = new FluidCell(this, depth+1, (index << 2) | 3, mass/4, width/2, height/2, temperature, hydrogen, helium, metals);
            se->setGravPotential(gravPotential);
            nw->sete(e);
            ne->sete(e);
            sw->sete(e);
            se->sete(e);
            nw->setE(E);
            ne->setE(E);
            sw->setE(E);
            se->setE(E);
            nw->setFusionEnergy(fusionE);
            ne->setFusionEnergy(fusionE);
            sw->setFusionEnergy(fusionE);
            se->setFusionEnergy(fusionE);

        }

        // FluidCell **getChildren() {
        //     return children;
        // }

        bool hasChildren() {
            return ((nw != nullptr) || (ne != nullptr) || (sw != nullptr) || (se != nullptr));
        }

        void absorbChildProps() {
            if (isLeaf()) return;
            FluidCell *children[4] {nw, ne, sw, se};
            long double totMass = 0; 
            long double tote = 0;
            long double totE = 0;
            long double totFusE = 0;
            long double avgX = 0;
            long double avgY = 0;
            long double avgZ = 0;
            double avgTemp = 0;
            double avgPressure = 0;
            double avgGravPotential = 0;

            for (int i = 0; i < 4; i++) {
                totMass += children[i]->getMass();
                tote += children[i]->gete(false);
                totE += children[i]->getE(false);
                totFusE += children[i]->getFusionEnergy()*children[i]->getMass();
                avgTemp += children[i]->getTemp(false);
                avgPressure += children[i]->getPressure(false);
                avgGravPotential += children[i]->getGravPotential();
                avgX += children[i]->getX()*children[i]->getMass();
                avgY += children[i]->getY()*children[i]->getMass();
                avgZ += children[i]->getZ()*children[i]->getMass();
            }

            avgTemp /= 4;
            avgPressure /= 4;
            avgGravPotential /= 4;
            tote /= 4;
            totE /= 4;
            totFusE /= 4;
            avgX /= totMass;
            avgY /= totMass;
            avgZ /= totMass;

            setMass(totMass);
            sete(tote);
            setE(totE);
            setFusionEnergy(totFusE);
            setTemp(avgTemp);
            setPressure(avgPressure);
            setGravPotential(avgGravPotential);
            setX(avgX);
            setY(avgY);
            setZ(avgZ);
        }

        void coarsen() {
            if (hasChildren()) {
                absorbChildProps();
                // delete nw;
                // delete ne;
                // delete sw;
                // delete se;
                nw = nullptr;
                ne = nullptr;
                sw = nullptr;
                se = nullptr;
            }
        }
	
        bool isLeaf() {
            return !(hasChildren());
        }

        FluidCell *nw, *ne, *sw, *se;
        FluidCell *parent;
        uint32_t depth, index;
        VelocityVector newVelocity;
        VelocityVector newVelocityx, newVelocityy;
        long double newMass, newE;
        long double newMassx, newMassy, newEx, newEy;
        float newX, newY, newZ;
        float newXx, newXy, newYx, newYy, newZx, newZy;
        float massODs, EODs; // overdraw scale
        bool shouldRefine = false;
        bool shouldCoarsen = false;
        bool refinedThisStep = false;
	    long double PG = 0;
    private:
        long double mass, e, E;
        double size, density, temperature, pressure, gravPotential, width, height, fusionE; 
        uint degenerate;
        int row, col;
        VelocityVector velocity;
        Neighbors neighbors;
        float hydrogen, helium, metals;
        // FluidCell *children[4] = {nw, ne, sw, se};
};

// A class to initialize a star in a FluidGrid
class Star {
    public:
	Star(double SolMass, double Radius, double rotation): totRadius(Radius) {
		totMass = SolMass*consts::SUN_MASS;
		// cenDensity = totMass/(4*consts::PI*8/105*pow(Radius,3));
		LEcoords = solveLaneEmden(0.0,1.0,0.0,0.1);
		//std::cout << LE["th"][0] << std::endl;
		int num_coords = LEcoords["xi"].size();
		xi1p = (LEcoords["th"][num_coords-1] - LEcoords["th"][num_coords-2])/(LEcoords["xi"][num_coords-1]-LEcoords["xi"][num_coords-2]); //dth/dxi(xi1)
																			double y_int = LEcoords["th"][num_coords-1] - xi1p*LEcoords["xi"][num_coords-1];
		xi1 = -y_int/xi1p; 
		std::cout << LEcoords["xi"][num_coords-1] << std::endl;
		// std::cout << "last coord: " << LEcoords["xi"][num_coords-1] << "," << LEcoords["th"][num_coords-1] << std::endl;
		double alph = Radius/xi1;
		cenDensity = totMass/(-4*consts::PI*pow(xi1,2)*xi1p*pow(alph,3));
		// rotation == surface velocity
	    }

	void placeInGrid(double x, double y, std::vector<FluidCell*> leaves, double width, double height) {
		for (int i = 0; i < leaves.size(); i++) {
			// x and y are offsets from center
			FluidCell *cell = leaves[i];
			xyPos xy = getXY(cell->depth, cell->index, width, height);
			double radius = pow(pow(width/2-x-xy.x,2) + pow(height/2-y-xy.y,2),0.5);
			if (radius < totRadius) {
				double density = getDensity(radius);
				double temperature = getTemperature(radius, cell->getCv());
				cell->setMass(density*cell->getSize());
				cell->setTemp(temperature);
				cell->gete(true);
				cell->setE(cell->gete(false));
				cell->getPressure(true);
			}
		}
		
	    }

	double getTemperature(double radius, double Cv) {
		double K = 4*consts::PI*consts::G*pow(totRadius/xi1,2)/(4*pow(cenDensity,1.0/3.0-1));	
		
		return K*consts::mH/consts::kb*pow(cenDensity,1.0/3.0)*getTheta(radius/totRadius*xi1);
		//return K/Cv*(1.4-1)*pow(cenDensity,1.0/3.0)*getTheta(radius/totRadius*xi1);
		
		//double pressure = 4*consts::PI*consts::G*pow(cenDensity,2)*((double) 13/317*pow(totRadius,2) + (double) (1/6*pow(radius,2) - 4/15*pow(radius,4)/pow(totRadius,2) \
														  + 67/315*pow(radius,6)/pow(totRadius,4) - 3/35*pow(radius,8)/pow(totRadius,6)\
																	  + 1/70*pow(radius,10)/pow(totRadius,8)));
		//if (radius == 0) {
		//	std::cout << "P, T: " << (double) 4*consts::PI*consts::G*pow(cenDensity,2)*13/317*pow(totRadius,2) << ", " << pressure/((1.4-1.0)*Cv*getDensity(radius)) << std::endl;
		//}
		//return pressure/((1.4-1.0)*Cv*getDensity(radius));
	    }

        double getDensity(double radius) {
	    return radius < totRadius ? cenDensity*pow(getTheta(radius/totRadius*xi1),3) : 1e-8;
		//return radius < totRadius ? cenDensity*pow((1 - pow(radius/totRadius, 2)), 2) : 1e-8;
	    }

            xyPos getXY(uint32_t depth, uint32_t index, double width, double height) {
            /* Gets position of bottom left corner of cell */
            uint32_t xmask = 0x1;
            uint32_t ymask = 0x2;
            uint16_t xInd = 0x0;
            uint16_t yInd = 0x0;
            for (int i = 0; i < depth; i++) {
                xInd = ((xmask & index) >> i) | xInd;
                xmask = (xmask << 2);
                yInd = ((ymask & index) >> (i+1)) | yInd;
                ymask = (ymask << 2);
            }
            double cellWidth = width / pow(pow(4,depth),0.5);
            double cellHeight = height / pow(pow(4,depth),0.5);
            xyPos xy;
            xy.x = cellWidth*xInd;
            xy.y = height - cellHeight*(yInd+1);
            return xy;
        }

	    std::map<std::string, std::vector<double>> solveLaneEmden(double xii, double thi, double yi, double h) {
			double xi0 = xii;
			double th0 = thi;
			double y0 = yi;
			std::map<std::string, std::vector<double>> coords = {{"xi", std::vector<double>{}}, {"th", std::vector<double>{}}};
			auto dy = [](double xi, double th, double y) { if (xi != 0.0f) return (double) -pow(th,3) - (double) 2/xi * y; else return (double) -1/3; };
			std::cout << h*dy(0.1,0.99,-0.1) << std::endl;
			while (th0 > 0) {
				double k1th = h*y0;
				double k1y = h*dy(xi0, th0, y0);
				double k2th = h*(y0+k1y/2);
				double k2y = h*dy(xi0+h/2, th0+k1th/2, y0+k1y/2);
				double k3th = h*(y0+k2y/2);
				double k3y = h*dy(xi0+h/2, th0+k2th/2, y0+k2y/2);
				double k4th = h*(y0+k3y);
				double k4y = h*dy(xi0+h, th0+k3th, y0+k3y);
				//std::cout << k4th << " " << std::flush;
				th0 = th0 + ((double) 1/(double) 6) * (k1th + 2*k2th + 2*k3th + k4th);
				y0 = y0 + ((double) 1/(double) 6) * (k1y + 2*k2y + 2*k3y + k4y);
				
				std::vector<double> vxi = coords["xi"];
				std::vector<double> vth = coords["th"];
				xi0 = xi0+h;
				vxi.push_back(xi0);
				vth.push_back(th0);
				coords["xi"] = vxi;
				coords["th"] = vth;
				//th0 = th1;
				//y0 = y1;
			}
			return coords;
	    }

	    std::map<std::string, std::vector<double>> solveLaneEmdenRec(double xi0, double th0, double y0, double h) {
			if (th0 > 0) {
				auto dy = [](double xi, double th, double y) { if (xi != 0) return -pow(th,3) - 2/xi * y; else return (double) -1/3; };
				double k1th = h*y0;
				double k1y = h*dy(xi0, th0, y0);
				double k2th = h*(y0+k1y/2);
				double k2y = h*dy(xi0+h/2, th0+k1th/2, y0+k1y/2);
				double k3th = h*(y0+k2y/2);
				double k3y = h*dy(xi0+h/2, th0+k2th/2, y0+k2y/2);
				double k4th = h*(y0+k3y);
				double k4y = h*(xi0+h, th0+k3th, y0+k3y);
				double th1 = th0 + 1/6 * (k1th + 2*k2th + 2*k3th + k4th);
				double y1 = y0 + 1/6 * (k1y + 2*k2y + 2*k3y + k4y);
				std::map<std::string,std::vector<double>> coords = solveLaneEmdenRec(xi0+h, th1, y1, h);
				std::vector<double> vxi = coords["xi"];
				std::vector<double> vth = coords["th"];
				vxi.push_back(xi0+h);
				vth.push_back(th1);
				coords["xi"] = vxi;
				coords["th"] = vth;
				return coords;
				
			} else {
				std::map<std::string, std::vector<double>> coords = {{"xi", std::vector<double>{}}, {"th", std::vector<double>{}}};
				return coords;
			}
	    }

	    double getTheta(double xi) {
		int loc;
		double dist = 1.0f; // minimum distance of xi from coordXi
		for (int i = 0; i < LEcoords["xi"].size(); i++) {
			double coordXi = LEcoords["xi"][i];
			double coordTheta = LEcoords["th"][i];
			if (abs(xi - coordXi) < dist) {
				dist = xi - coordXi;
				loc = i;
			}
		}

		int nextLoc;
		if (dist > 0) {
			nextLoc = loc + 1;
		} else if (dist < 0) {
			nextLoc = loc - 1;
		} else {
			return LEcoords["th"][loc];
		}

		double m = (LEcoords["th"][nextLoc] - LEcoords["th"][loc])/(LEcoords["xi"][nextLoc] - LEcoords["xi"][loc]);
		double b = LEcoords["th"][nextLoc] - m*LEcoords["xi"][nextLoc];
		return m*xi + b;
	    }
	
    private:
		double totMass, totRadius, cenDensity;
		std::map<std::string, std::vector<double>> LEcoords; 
		double xi1, xi1p;
};

class FluidGrid {
    public:
        FluidGrid(double width, double height, int startDepth, float dt): width(width), height(height), dt(dt) {
            root = new FluidCell(nullptr, 0, 0, 10, width, height, 1, 0.8, 0.15, 0.05);
            int numCells = refineGridtoDepth(root, 0, startDepth);
            assignIDs(root, 0, 0);
            leafCells.reserve(pow(4,maxDepth));
            updateLeafCells();
            setNeighbors();

		for (int i = 0; i < leafCells.size(); i++) {
			leafCells[i]->setMass(1e-8*leafCells[i]->getSize());
			leafCells[i]->setTemp(10);
		}
		
		Star *s1 = new Star(1.0, 6e8, 0);

		s1->placeInGrid(0, 0, leafCells, width, height);
			
            double mass, temp;
            // maxV = 0;
            // for (int i = 0; i < leafCells.size(); i++) {
            //     FluidCell *cell = leafCells[i];
            //     // mass = (std::rand() % 100) * 1e16;
            //     xyPos xy = getXY(cell->depth,cell->index);
            //     double radius = pow(width*height,0.5)/4;
            //     double dist = pow(pow(width/2-5e8-xy.x,2) + pow(height/2-3e8-xy.y,2),0.5);
            //     double Tc = 2e6;
            //     // double Tc = 1e8;
            //     long double mc = 3e21;
            //     // mass = dist < radius ? pow(1-dist/radius,3)*1e24/numCells : 1e18/numCells;
            //     mass = dist <= radius ? std::sin(consts::PI/2 * dist/radius)/(consts::PI *dist/radius) * mc/numCells : 1e15/numCells;
			// 		//1e-8*cell->getSize();
            //     // temp = dist < radius ? pow(1-dist/radius,3)*1e6 : 10;
            //     // temp = dist < radius ? std::cos(consts::PI/2 * dist / (radius*1.5)) / (consts::PI * dist / (radius*1.5)) * 1e5 : 10;
            //     temp = dist <= radius ? std::sin(consts::PI/2 * dist / radius) / (consts::PI * dist / radius) * Tc : 10;
            //     // temp = 1e6;

            //     VelocityVector vel;
            //     double xDist = xy.x -5e8 - width/2;
            //     double yDist = xy.y -3e8 - height/2;
            //     float angle = xDist > 0 ? std::atan(yDist/xDist) + consts::PI/2 : consts::PI+std::atan(yDist/xDist) + consts::PI/2;
            //     double velMag = std::sqrt(2*consts::G*1e27/dist);
            //     double vx = velMag*std::cos(angle);
            //     double vy = velMag*std::sin(angle);
            //     vel.setVx(vx);
            //     vel.setVy(vy);

            //     if (dist > radius || dist == 0) {
            //         if (dist == 0) {
            //             mass = mc/numCells;
            //             temp = Tc;
            //         }
            //         vel = VelocityVector(0,0);
            //     }

            //     cell->setVelocity(vel);
            //     // if (dist/radius < 1 && vy < 0) std::cout << " dist,vx,vy " << dist/radius << "," << cell->getVelocity().getVx() << "," << cell->getVelocity().getVy();
            //     cell->setMass(mass);
            //     cell->setTemp(temp);
            //     cell->gete(true);
            //     cell->setE(cell->gete(false));
            //     cell->getPressure(true);
            //     if (abs(cell->getVelocity().getVx()) > maxV) maxV = abs(cell->getVelocity().getVx());
            //     else if (abs(cell->getVelocity().getVy()) > maxV) maxV = abs(cell->getVelocity().getVy());
            // }
            
            minSize = std::min(width,height) / pow(pow(4,startDepth),0.5);
            // if (maxV == 0) {
                maxV = 0.4 * minSize/getdt();
            // }
        }

        int refineGridtoDepth(FluidCell *cell, int depth, int mDepth) {
            if (depth >= mDepth) {
                return 1;
            } else {
                cell->refine();
                FluidCell *children[4] {cell->nw, cell->ne, cell->sw, cell->se};
                int numCells = 0;
                for (int i = 0; i < 4; i++) {
                    numCells += refineGridtoDepth(children[i], depth+1, mDepth);
                }
                return numCells;
            }
        }

        FluidCell *adjacentCell(int depth, int index, Direction dir) {
            return adjacentCellHelper(depth, index, dir, depth, index);
        }

        FluidCell *adjacentCell(FluidCell *cell, Direction dir) {
            TreeLoc loc = getLocFromID(getIDfromLeaf(cell));
            int depth = loc.depth;
            int index = loc.index;
            FluidCell *adj = adjacentCellHelper(depth, index, dir, depth, index);
            if (adj == 0) return cell;
            else return adj;
        }

        FluidCell *adjacentCellHelper(int depth, int index, Direction dir, int origDepth, int origIndex) {
            // If depth <= 0, we are at a boundary
            if (depth <= 0) return 0;
            uint64_t id = getID(depth, index);
            uint idx = id & 0x3; // between 0 and 3 (0:nw, 3:se)
            int nextIdx = -1;
            switch (dir){
                case LEFT:
                    if (idx & 0x1) nextIdx = idx - 1;
                    break;
                case RIGHT:
                    if (!(idx & 0x1)) nextIdx = idx + 1;
                    break;
                case UP:
                    if ((idx >> 1) & 0x1) nextIdx = idx - 2;
                    break;
                case DOWN:
                    if (!((idx >> 1) & 0x1)) nextIdx = idx + 2;
            }

            if (nextIdx != -1) {
                // hide two lowest bits and set them to 0 and then add nextIdx
                uint64_t nextID = getID(depth, ((index >> 2) << 2) | nextIdx);
                Direction opDir;
                FluidCell *currCell;
                FluidCell *cell;
                int vecIdx;// to choose which candidate to use
                // if we are at an internal node
                if (!idToCell.count(nextID)) {
                    
                    switch (dir){
                        case LEFT:
                            opDir = RIGHT;
                            if ((origIndex >> 1) & 0x1) vecIdx = 1;
                            else vecIdx = 0; 
                            break;
                        case RIGHT:
                            opDir = LEFT;
                            if ((origIndex >> 1) & 0x1) vecIdx = 1;
                            else vecIdx = 0;
                            break;
                        case UP:
                            opDir = DOWN;
                            if (origIndex & 0x1) vecIdx = 1;
                            else vecIdx = 0;
                            break;
                        case DOWN:
                            opDir = UP;
                            if (origIndex & 0x1) vecIdx = 1;
                            else vecIdx = 0;
                    }

					uint32_t childIdx = ((uint32_t) nextID << 2) & 0xFFFFFFFF;
                    int childDepth = depth+1;
                    uint64_t childID = (((uint64_t) childDepth) << 32) | (uint64_t) childIdx;
                    int nDescended = 1;
                    // while we are not at a leaf, find the ID of a leaf in this
                    //// tree
                    while (!idToCell.count(childID)) {
                        childIdx = ((uint32_t) childIdx << 2) & 0xFFFFFFFF;
                        childDepth += 1;
                        childID = (((uint64_t) childDepth) << 32) | (uint64_t) childIdx;
                        nDescended += 1;
                    }
                    currCell = idToCell[childID];
                    for (int i = 0; i < nDescended; i++) {
                        currCell = currCell->parent;
                    }

                    cell = traverseInDirection(currCell,opDir,depth,origDepth, origIndex);
                    // This would normally happen with getIDfromLeaf but 
                    //// I wanted to hide the unnecessary warning about 
                    //// non-leafage.
                    if (cell->isLeaf()) nextID = getIDfromLeaf(cell); 
                    else nextID = 0;
                }
                uint32_t leafDepth = (nextID >> 32) & 0xFFFFFFFF;
                // if the resolutions are the same or neighbor is more coarse
                if (nextID == 0) {
                    return cell;
                } else if ((leafDepth == origDepth) || (leafDepth == origDepth - 1) || (leafDepth == origDepth + 1)) {
                    return idToCell[nextID];
                } else {
                    return nullptr;
                }
                
            } else {
                return adjacentCellHelper(depth - 1, index >> 2, dir, origDepth, origIndex);
            }

        }

        FluidCell *traverseInDirection(FluidCell *cell, Direction dir, int depth, int origDepth, int origIdx) {
            /* This will only go as deep as the orignal (target) depth and try to follow the origIdx's location. */
            if (cell->isLeaf() || (depth == origDepth)) {
                return cell;
            }
            uint16_t idx;
            switch (dir){
                case LEFT:
                    if ((origIdx >> (2*(origDepth-depth-1)) >> 1) & 0x1) {
                        return traverseInDirection(cell->sw,dir,depth+1,origDepth,origIdx);
                    } else {
                        return traverseInDirection(cell->nw,dir,depth+1,origDepth,origIdx);
                    }
                    break;
                case RIGHT:
                    if ((origIdx >> (2*(origDepth-depth-1)) >> 1) & 0x1) {
                        return traverseInDirection(cell->se,dir,depth+1,origDepth,origIdx);
                    } else {
                        return traverseInDirection(cell->ne,dir,depth+1,origDepth,origIdx);
                    }
                    break;
                case UP:
                    if (origIdx >> (2*(origDepth-depth-1)) & 0x1) {
                        return traverseInDirection(cell->ne,dir,depth+1,origDepth,origIdx);
                    } else {
                        return traverseInDirection(cell->nw,dir,depth+1,origDepth,origIdx);
                    }
                    break;
                case DOWN:
                    if (origIdx >> (2*(origDepth-depth-1)) & 0x1) {
                        return traverseInDirection(cell->se,dir,depth+1,origDepth,origIdx);
                    } else {
                        return traverseInDirection(cell->sw,dir,depth+1,origDepth,origIdx);
                    }
            }
            return nullptr;
        }

        uint64_t getIDfromLeaf(FluidCell *cell) {
            return getID(cell->depth, cell->index);
        }

        TreeLoc getLocFromID(uint64_t id) {
            uint32_t index = id & 0xFFFFFFFF;
            uint32_t depth = (id >> 32) & 0xFFFFFFFF;
            TreeLoc loc;
            loc.index = index;
            loc.depth = depth;
            return loc;
        }

        void setNeighbors() {
            uint32_t depth;
            uint32_t index;
            FluidCell *top, *bottom, *left, *right;
            FluidCell *cell;
            for (const auto& pair : idToCell) {
                TreeLoc loc = getLocFromID(pair.first);
                depth = loc.depth;
                index = loc.index;

                top = adjacentCell(depth,index,UP);
                if (top == 0) top = pair.second;
                pair.second->setTop(top);
                
                bottom = adjacentCell(depth,index,DOWN);
                if (bottom == 0) bottom = pair.second;
                pair.second->setBottom(bottom);
                
                left = adjacentCell(depth,index,LEFT);
                if (left == 0) left = pair.second;
                pair.second->setLeft(left);
                
                right = adjacentCell(depth,index,RIGHT);
                if (right == 0) right = pair.second;
                pair.second->setRight(right);
                
            }

        }

        uint64_t getID(int depth, int index) {
            // std::cout << depth << ", " << index << std::endl;
            assert(depth >= 0);
            assert(index < pow(4,depth));
            return ((uint64_t) depth << 32) | index;
        }

        void assignIDs(FluidCell *cell, int depth, int index) {
            uint64_t id = getID(depth, index);
            if (cell != nullptr) {
                FluidCell *children[4] {cell->nw, cell->ne, cell->sw, cell->se};
                if (cell->isLeaf()) idToCell[id] = cell;
                if (cell->hasChildren()) idToCell.erase(id);
                for (int i = 0; i < 4; i++) {
                    assignIDs(children[i], depth+1, (index << 2 | i));
                }
            } else {
                idToCell.erase(id);
            }
        }

        void printID(uint64_t num) {
            if (num > 1) {
                printID(num / 2);
            }
            std::cout << (num % 2);
        }

        void refineCell(int depth, int index) {
            uint64_t id = getID(depth, index);
            assert(idToCell.count(id)); // Can only refine leaf nodes
            FluidCell *cell = idToCell[id];
            cell->refine();
            // make sure any potential jumps get resolved
            if (cell->depth > cell->getRight()->depth && !cell->getRight()->refinedThisStep) refineCell(cell->getRight()->depth, cell->getRight()->index);
            if (cell->depth > cell->getLeft()->depth && !cell->getLeft()->refinedThisStep) refineCell(cell->getLeft()->depth, cell->getLeft()->index);
            if (cell->depth > cell->getTop()->depth && !cell->getTop()->refinedThisStep) refineCell(cell->getTop()->depth, cell->getTop()->index);
            if (cell->depth > cell->getBottom()->depth && !cell->getBottom()->refinedThisStep) refineCell(cell->getBottom()->depth, cell->getBottom()->index);
            // cell->refinedThisStep = true;
            
            // assignIDs(cell, depth, index);
            // this is faster than recursive call
            idToCell.erase(getID(cell->depth, cell->index));
            idToCell[getID(cell->nw->depth, cell->nw->index)] = cell->nw;
            idToCell[getID(cell->ne->depth, cell->ne->index)] = cell->ne;
            idToCell[getID(cell->sw->depth, cell->sw->index)] = cell->sw;
            idToCell[getID(cell->se->depth, cell->se->index)] = cell->se;
            
            if (cell->hasChildren()) {
                if (cell->nw->getWidth() < minSize) minSize = cell->nw->getWidth();
                else if (cell->nw->getHeight() < minSize) minSize = cell->nw->getHeight();
            }
            
            
        }

        void coarsenCell(int depth, int index) {
            // Get ID of cell to coarsen
            uint64_t parentID = getID(depth, index);
            // Index of first (nw) child will be 4 times index of parent
            uint32_t childIndex = ((uint32_t) parentID << 2);
            // Child is in next layer—get the ID
            uint64_t childID = getID(depth+1, childIndex);
            // Find child in map and grab it's parent to coarsen
            FluidCell *child = idToCell[childID];
            child->parent->coarsen();
            assignIDs(child->parent, depth, index);
        }
    
        std::map<uint64_t,FluidCell*> getIDMap() {
            return idToCell;
        }

        uint64_t getIDFromXY(double x, double y) {
            // returns the ID based on the current max depth
            double cellWidth, cellHeight;
            uint32_t ID = 0;
            uint32_t col, row;
            char mask = 0x1;
            cellWidth = width / pow(pow(4,maxDepth),0.5);
            cellHeight = height / pow(pow(4,maxDepth),0.5);
            col = (int)(x/cellWidth);
            row = (int)((height - y)/cellHeight);
            for (int i = 0; i < maxDepth; i++) {
                ID = ID | (((col & mask) | ((row & mask) << 1)) << i);
                mask = mask << 1;
            }
            return ((uint64_t) maxDepth<<32) + ID;
        }

        uint64_t getIDFromXY(position pos) {
            // returns the ID based on the current max depth
            double x = pos.x;
            double y = pos.y;
            double cellWidth, cellHeight;
            uint32_t ID = 0;
            uint32_t col, row;
            char mask = 0x1;
            cellWidth = width / pow(pow(4,maxDepth),0.5);
            cellHeight = height / pow(pow(4,maxDepth),0.5);
            col = (int)(x/cellWidth);
            row = (int)((height - y)/cellHeight);
            for (int i = 0; i < maxDepth; i++) {
                ID = ID | (((col & mask) | ((row & mask) << 1)) << i);
                mask = mask << 1;
            }
            return ((uint64_t) maxDepth<<32) + ID;
        }

        FluidCell *getCellFromXY(double x, double y) {
            uint64_t ID = getIDFromXY(x, y);
            uint32_t depth = ID >> 32;
            uint32_t index = ID & 0xffffffff;
            while (!idToCell.count(ID)) {
                ID = ((uint64_t) (depth - 1) << 32) | (index >> 2);
            }
            return idToCell[ID];
        }
        
        xyPos getXY(uint32_t depth, uint32_t index) {
            /* Gets position of bottom left corner of cell */
            uint32_t xmask = 0x1;
            uint32_t ymask = 0x2;
            uint16_t xInd = 0x0;
            uint16_t yInd = 0x0;
            for (int i = 0; i < depth; i++) {
                xInd = ((xmask & index) >> i) | xInd;
                xmask = (xmask << 2);
                yInd = ((ymask & index) >> (i+1)) | yInd;
                ymask = (ymask << 2);
            }
            double cellWidth = width / pow(pow(4,depth),0.5);
            double cellHeight = height / pow(pow(4,depth),0.5);
            xyPos xy;
            xy.x = cellWidth*xInd;
            xy.y = height - cellHeight*(yInd+1);
            return xy;
        }

        FluidCell *getRoot() {
            return root;
        }

        float getdt() {
            return dt;
        }

        void solveGravPotential(int iters) {
            int n;
            for (n = 0; n < iters; n++) {
                for (int i = 0; i < leafCells.size(); i++) {
                    double uL, uR, uT, uB = 0;
                    FluidCell *cell = leafCells[i];
                    if ((cell->getTop() == cell) || (cell->getBottom() == cell) || (cell->getLeft() == cell) || (cell->getRight() == cell)) {
                        cell->setGravPotential(0);
                        continue;
                    }
                    double dens = cell->getDensity();
                    double w2 = pow(cell->getWidth(),2);
                    double h2 = pow(cell->getHeight(),2);
                    cell->getTop()->absorbChildProps();
                    cell->getBottom()->absorbChildProps();
                    cell->getLeft()->absorbChildProps();
                    cell->getRight()->absorbChildProps();
                    uT = cell->getTop()->getGravPotential();
                    uB = cell->getBottom()->getGravPotential();
                    uL = cell->getLeft()->getGravPotential();
                    uR = cell->getRight()->getGravPotential();
                    double u = (uL + uR)/(2*(1+w2/h2)) + (uT + uB)/(2*(h2/w2+1)) - 2 * consts::PI * consts::G * dens / (1/w2 + 1/h2);
                    cell->setGravPotential(u);
                }
                
            }
        }

        void updateVelocities() {
            for (int i = 0; i < leafCells.size(); i++) {
                FluidCell *cell = leafCells[i];
                cell->setE(cell->gete(false) + 0.5*cell->getDensity()*cell->getVelocity().getMag()*cell->getVelocity().getMag());
                FluidCell *cellT = cell->getTop();
                FluidCell *cellB = cell->getBottom();
                FluidCell *cellL = cell->getLeft();
                FluidCell *cellR = cell->getRight();
                VelocityVector newV = VelocityVector(0,0);
                double gravPotT, gravPotB, gravPotL, gravPotR;
                double pressT, pressB, pressL, pressR;
                double distT = (cell->getHeight()/2 + cellT->getHeight()/2);
                double distB = (cell->getHeight()/2 + cellB->getHeight()/2);
                double distL = (cell->getWidth()/2 + cellL->getWidth()/2);
                double distR = (cell->getWidth()/2 + cellR->getWidth()/2);
                if (cellT->hasChildren()) {
                    gravPotT = (cellT->sw->getGravPotential() + cellT->se->getGravPotential())/2;
                    pressT = (cellT->sw->getPressure(true) + cellT->se->getPressure(true))/2;
                    distT = (cell->getHeight()/2 + cellT->sw->getHeight()/2);
                } else {
                    gravPotT = cellT->getGravPotential();
                    pressT = cellT->getPressure(true);
                }

                if (cellB->hasChildren()) {
                    gravPotB = (cellB->nw->getGravPotential() + cellB->ne->getGravPotential())/2;
                    pressB = (cellB->nw->getPressure(true) + cellB->ne->getPressure(true))/2;
                    distB = (cell->getHeight()/2 + cellB->nw->getHeight()/2);
                } else {
                    gravPotB = cellB->getGravPotential();
                    pressB = cellB->getPressure(true);
                }

                if (cellL->hasChildren()) {
                    gravPotL = (cellL->ne->getGravPotential() + cellL->se->getGravPotential())/2;
                    pressL = (cellL->ne->getPressure(true) + cellL->se->getPressure(true))/2;
                    distL = (cell->getWidth()/2 + cellL->ne->getWidth()/2);
                } else {
                    gravPotL = cellL->getGravPotential();
                    pressL = cellL->getPressure(true);
                }

                if (cellR->hasChildren()) {
                    gravPotR = (cellR->nw->getGravPotential() + cellR->sw->getGravPotential())/2;
                    pressR = (cellR->nw->getPressure(true) + cellR->sw->getPressure(true))/2;
                    distR = (cell->getWidth()/2 + cellR->nw->getWidth()/2);
                } else {
                    gravPotR = cellR->getGravPotential();
                    pressR = cellR->getPressure(true);
                }

                double gradUy = (gravPotT - gravPotB)/(distT+distB);
                double gradUx = (gravPotR - gravPotL)/(distL+distR);
                double gradPy = (pressT - pressB)/(distT+distB);
                double gradPx = (pressR - pressL)/(distL+distR);
                VelocityVector v = cell->getVelocity();
                if (cell->getDensity() > 0) {
                    VelocityVector diff = velDiffusion(cell);
					double avgXDens = (cellL->getDensity() + cell->getDensity() + cellR->getDensity())/3;
					double avgYDens = (cellT->getDensity() + cell->getDensity() + cellB->getDensity())/3;
					cell->pressGrad((abs(gradPx)/avgXDens + abs(gradPy)/avgYDens)/2);

					double newVx = v.getVx() + (-gradUx-gradPx/avgXDens + diff.getVx()/cell->getDensity())*getdt();
                    double newVy = v.getVy() + (-gradUy-gradPy/avgYDens + diff.getVy()/cell->getDensity())*getdt();
                    xyPos pos = getXY(cell->depth, cell->index);

                    double scaleX, scaleY;
                    if (newVx > 0) {
                        scaleX = newVx*getdt()/pos.x > 1 ? pos.x/(-newVx*getdt()) : 1;
                    } else {
                        scaleX = -newVx*getdt()/(width-pos.x) > 1 ? (width-pos.x)/(newVx*getdt()) : 1;
                    }

                    if (newVy > 0) {
                        scaleY = newVy*getdt()/pos.y > 1 ? pos.y/(-newVy*getdt()) : 1;
                    } else {
                        scaleY = -newVy*getdt()/(height-pos.y) > 1 ? (height-pos.y)/(newVy*getdt()) : 1;
                    }

                    if (!std::isfinite(newVx+newVy)) {
                        newVx = v.getVx();
                        newVy = v.getVy();
                    }

                    newV.setVx(newVx*scaleX);
                    newV.setVy(newVy*scaleY);
                } else {
					// get average of surroundings
					double Rvx = cellR != cell ? cellR->getVelocity().getVx() : 0;
					double Lvx = cellL != cell ? cellL->getVelocity().getVx() : 0;
					double Tvx = cellT != cell ? cellT->getVelocity().getVx() : 0;
					double Bvx = cellB != cell ? cellB->getVelocity().getVx() : 0;
					double newVx = (Rvx + Lvx + Tvx + Bvx)/4;
					double Rvy = cellR != cell ? cellR->getVelocity().getVy() : 0;
					double Lvy = cellL != cell ? cellL->getVelocity().getVy() : 0;
					double Tvy = cellT != cell ? cellT->getVelocity().getVy() : 0;
					double Bvy = cellB != cell ? cellB->getVelocity().getVy() : 0;
					double newVy = (Rvy + Lvy + Tvy + Bvy)/4;

                    newV.setVx(newVx);
                    newV.setVy(newVy);
                }

                if (cell == cellR || cell == cellL) {
                    newV.setVx(0);
                }
                if (cell == cellT || cell == cellB) {
                    newV.setVy(0);
                }

                if (newV.getMag()*getdt()/cell->getWidth() > 4) {
                    long double factor = 4/(newV.getMag()*getdt()/cell->getWidth());
                    newV.setVx(newV.getVx()*factor);
                    newV.setVy(newV.getVy()*factor);
                }

                cell->setVelocity(newV);
            }
        }

        VelocityVector velDiffusion(FluidCell *cell) {
            FluidCell *cellT = cell->getTop();
            FluidCell *cellB = cell->getBottom();
            FluidCell *cellL = cell->getLeft();
            FluidCell *cellR = cell->getRight();
            
            double distT = (cell->getHeight()/2 + cellT->getHeight()/2);
            double distB = (cell->getHeight()/2 + cellB->getHeight()/2);
            double distL = (cell->getWidth()/2 + cellL->getWidth()/2);
            double distR = (cell->getWidth()/2 + cellR->getWidth()/2);

            VelocityVector v = cell->getVelocity();
            VelocityVector vR = cellR->getVelocity();
            VelocityVector vL = cellL->getVelocity();
            VelocityVector vB = cellB->getVelocity();
            VelocityVector vT = cellT->getVelocity();

            VelocityVector l,r;

            if (cellT->hasChildren()) {
                distT = (cell->getHeight()/2 + cellT->sw->getHeight()/2);
                l = cellT->sw->getVelocity();
                r = cellT->se->getVelocity();
                vT = (l + r)/2;
            }

            if (cellB->hasChildren()) {
                distB = (cell->getHeight()/2 + cellB->nw->getHeight()/2);
                l = cellB->nw->getVelocity();
                r = cellB->ne->getVelocity();
                vB = (l + r)/2;
            }

            if (cellL->hasChildren()) {
                distL = (cell->getWidth()/2 + cellL->ne->getWidth()/2);
                l = cellL->ne->getVelocity();
                r = cellL->se->getVelocity();
                vL = (l + r)/2;
            }

            if (cellR->hasChildren()) {
                distR = (cell->getWidth()/2 + cellR->nw->getWidth()/2);
                l = cellR->nw->getVelocity();
                r = cellR->sw->getVelocity();
                vR = (l + r)/2;
            }

            VelocityVector dvR2 = (vR - v) / distR;
            VelocityVector dvL2 = (v - vL) / distL;
            VelocityVector dvT2 = (vT - v) / distT;
            VelocityVector dvB2 = (v - vB) / distB;
            VelocityVector divX = (dvR2 - dvL2)/(distL+distR);
            VelocityVector divY = (dvT2 - dvB2)/(distT+distB);

            return std::isfinite((divX+divY).getVx()) && std::isfinite((divX+divY).getVy()) ? (divX + divY)*viscosity : VelocityVector(0,0);

        }

        void energyUpdate() {
            for (int i = 0; i < leafCells.size(); i++) {
                FluidCell *cell = leafCells[i];
                float energyReduction = 1;
                if (cell->getDensity() < 1e-6) {
                    continue;
                }
                FluidCell *cellT = cell->getTop();
                FluidCell *cellB = cell->getBottom();
                FluidCell *cellL = cell->getLeft();
                FluidCell *cellR = cell->getRight();

                double density = cell->getDensity();

                double gravT, gravB, gravL, gravR;
                double pressT, pressB, pressL, pressR;
                VelocityVector vT, vB, vL, vR;
                double distT = (cell->getHeight()/2 + cellT->getHeight()/2);
                double distB = (cell->getHeight()/2 + cellB->getHeight()/2);
                double distL = (cell->getWidth()/2 + cellL->getWidth()/2);
                double distR = (cell->getWidth()/2 + cellR->getWidth()/2);
                VelocityVector l,r;

                if (cellT->hasChildren()) {
                    gravT = (cellT->sw->getGravPotential() + cellT->se->getGravPotential())/2;
                    pressT = (cellT->sw->getPressure(true) + cellT->se->getPressure(true))/2;
                    l = cellT->sw->getVelocity();
                    r = cellT->se->getVelocity();
                    vT = (l + r)/2;
                    distT = (cell->getHeight()/2 + cellT->sw->getHeight()/2);
                } else {
                    gravT = cellT->getGravPotential();
                    pressT = cellT->getPressure(true);
                    vT = cellT->getVelocity();
                }

                if (cellB->hasChildren()) {
                    gravB = (cellB->nw->getGravPotential() + cellB->ne->getGravPotential())/2;
                    pressB = (cellB->nw->getPressure(true) + cellB->ne->getPressure(true))/2;
                    l = cellB->nw->getVelocity();
                    r = cellB->ne->getVelocity();
                    vB = (l + r)/2;
                    distB = (cell->getHeight()/2 + cellB->nw->getHeight()/2);
                } else {
                    gravB = cellB->getGravPotential();
                    pressB = cellB->getPressure(true);
                    vB = cellB->getVelocity();
                }

                if (cellL->hasChildren()) {
                    gravL = (cellL->ne->getGravPotential() + cellL->se->getGravPotential())/2;
                    pressL = (cellL->ne->getPressure(true) + cellL->se->getPressure(true))/2;
                    l = cellL->ne->getVelocity();
                    r = cellL->se->getVelocity();
                    vL = (l + r)/2;
                    distL = (cell->getWidth()/2 + cellL->ne->getWidth()/2);
                } else {
                    gravL = cellL->getGravPotential();
                    pressL = cellL->getPressure(true);
                    vL = cellL->getVelocity();
                }

                if (cellR->hasChildren()) {
                    gravR = (cellR->nw->getGravPotential() + cellR->sw->getGravPotential())/2;
                    pressR = (cellR->nw->getPressure(true) + cellR->sw->getPressure(true))/2;
                    l = cellR->nw->getVelocity();
                    r = cellR->sw->getVelocity();
                    vR = (l + r)/2;
                    distR = (cell->getWidth()/2 + cellR->nw->getWidth()/2);
                } else {
                    gravR = cellR->getGravPotential();
                    pressR = cellR->getPressure(true);
                    vR = cellR->getVelocity();
                }
                long double mass = cell->getMass();

                double pressure = cell->getPressure();
                double E = cell->getE(false);

                // nuclear fusion
                double fusionEnergy = fusionRateH(density, cell->getTemp(false), cell->getX()) * getdt();
                double fusionHeEnergy = fusionRateHe(density, cell->getTemp(false), cell->getY()) * getdt();

                cell->setFusionEnergy((fusionEnergy+fusionHeEnergy)/getdt());

                // Update hydrogen and helium abundances
                double dX = std::min((double)1.0,fusionEnergy / (consts::QH_He * density));
                cell->HtoHe(dX);
                double dY = std::min((double)1.0,fusionHeEnergy / (consts::QHe_C * density));
                cell->HetoZ(dY);

                if (dX > 1) fusionEnergy = consts::QH_He * density;
                if (dY > 1) fusionHeEnergy = consts::QHe_C * density;

                E += energyReduction*(fusionEnergy+fusionHeEnergy);

                // pressure-work term
                if (cell == cellR) {
                    vR.setVx(vL.getVx());
                }
                if (cell == cellL) {
                    vL.setVx(vR.getVx());
                }
                if (cell == cellT) {
                    vT.setVy(vB.getVy());
                }
                if (cell == cellB) {
                    vB.setVy(vT.getVy());
                }
				
                double div = (vR.getVx()-vL.getVx())/(distR+distL) + (vT.getVy()-vB.getVy())/(distT+distB);
                double pdiv = pressure * div;
                if (cell == cellR || cell == cellL || cell == cellT || cell == cellB) {
                    pdiv = 0.0f;
                }
                if (E - pdiv * getdt() >= 0) {
                    E += energyReduction*(-pdiv * getdt());
                } else {
                    E = 0.0f; // Prevent negative energy
                }

                // cooling
                double T = cell->getTemp(false);
                double Z = cell->getZ();
                double lambda = coolingFunction(density,T,Z);
                if (E - lambda * getdt() >= 0) {
                    E += -lambda*getdt();
                } else {
                    E = 0.0f;
                }

                // gravitational-work term
                double grav = cell->getGravPotential();
                double gy = -(gravT-gravB)/(distT+distB);
                double gx = -(gravR-gravL)/(distR+distL);
                double gravWork = (cell->getVelocity().getVx()*gx+cell->getVelocity().getVy()*gy)*density;
                if (cell == cellR || cell == cellL || cell == cellT || cell == cellB) {
                    gravWork = 0.0f;
                }

                if (E + gravWork * getdt() >= 0) {
                    E += energyReduction*(gravWork * getdt());
                } else {
                    // std::cerr << "Warning: Gravitational work term causing negative energy.\n";
                    E = 0.0f; // Prevent negative energy
                }

                cell->setE(E);
                
            }
        }

        long double radiativeTransfer(FluidCell *cell) {
            FluidCell *cellT = cell->getTop();
            FluidCell *cellB = cell->getBottom();
            FluidCell *cellL = cell->getLeft();
            FluidCell *cellR = cell->getRight();

            double density = cell->getDensity();
            double densityT = cellT->getDensity();
            double densityB = cellB->getDensity();
            double densityL = cellL->getDensity();
            double densityR = cellR->getDensity();

            double X = cell->getX();
            double XT = cellT->getX();
            double XB = cellB->getX();
            double XL = cellL->getX();
            double XR = cellR->getX();

            double temp = cell->getTemp(false);
            double tempT = cellT->getTemp(false);
            double tempB = cellB->getTemp(false);
            double tempL = cellL->getTemp(false);
            double tempR = cellR->getTemp(false);

            double E = cell->getE(false);
            double ET = cellT->getE(false);
            double EB = cellB->getE(false);
            double EL = cellL->getE(false);
            double ER = cellR->getE(false);

            double distT = (cell->getHeight()/2 + cellT->getHeight()/2);
            double distB = (cell->getHeight()/2 + cellB->getHeight()/2);
            double distL = (cell->getWidth()/2 + cellL->getWidth()/2);
            double distR = (cell->getWidth()/2 + cellR->getWidth()/2);



            if (cellT->hasChildren()) {
                densityT = (cellT->sw->getDensity() + cellT->se->getDensity())/2;
                tempT = (cellT->sw->getTemp(false) + cellT->se->getTemp(false))/2;
                ET = (cellT->sw->getE(false) + cellT->se->getE(false))/2;
                XT = (cellT->sw->getX() + cellT->se->getX())/2;
                distT = (cell->getHeight()/2 + cellT->sw->getHeight()/2);
            }

            if (cellB->hasChildren()) {
                densityB = (cellB->nw->getDensity() + cellB->ne->getDensity())/2;
                tempB = (cellB->nw->getTemp(false) + cellB->ne->getTemp(false))/2;
                EB = (cellB->nw->getE(false) + cellB->ne->getE(false))/2;
                XB = (cellB->nw->getX() + cellB->ne->getX())/2;
                distB = (cell->getHeight()/2 + cellB->nw->getHeight()/2);
            }

            if (cellL->hasChildren()) {
                densityL = (cellL->ne->getDensity() + cellL->se->getDensity())/2;
                tempL = (cellL->ne->getTemp(false) + cellL->se->getTemp(false))/2;
                EL = (cellL->ne->getE(false) + cellL->se->getE(false))/2;
                XL = (cellL->ne->getX() + cellL->se->getX())/2;
                distL = (cell->getWidth()/2 + cellL->ne->getWidth()/2);
            }

            if (cellR->hasChildren()) {
                densityR = (cellR->nw->getDensity() + cellR->sw->getDensity())/2;
                tempR = (cellR->nw->getTemp(false) + cellR->sw->getTemp(false))/2;
                ER = (cellR->nw->getE(false) + cellR->sw->getE(false))/2;
                XR = (cellR->nw->getX() + cellR->sw->getX())/2;
                distR = (cell->getWidth()/2 + cellR->nw->getWidth()/2);
            } 

            long double D, Dr, Dl, Dt, Db, Dr2, Dl2, Dt2, Db2;
            double kap, kapR, kapL, kapT, kapB, tau, tauR, tauL, tauT, tauB;

            kap = opacity(density,temp,X);
            kapR = opacity(densityR,tempR,XR);
            kapL = opacity(densityL,tempL,XL);
            kapT = opacity(densityT,tempT,XT);
            kapB = opacity(densityB,tempB,XB);
            tau = opticalDepth(kap,density,cell->getWidth());
            tauR = opticalDepth(kapR,densityR,cellR->getWidth());
            tauL = opticalDepth(kapL,densityL,cellL->getWidth());
            tauT = opticalDepth(kapT,densityT,cellT->getHeight());
            tauB = opticalDepth(kapB,densityB,cellB->getHeight());

            // double gradE = std::sqrt( pow((ER-EL)/(distR+distL),2) + pow((ET-EB)/(distT+distB),2) );

            // double R = gradE/(kap*density*E);
            // double lam = 1/R * (1/std::tanh(R) - 1/R);

            // cell-centered D
            D = (4/3*consts::a*consts::c*pow(temp,3)/(kap*density))/(1+pow(tau,-2));
            Dr = (4/3*consts::a*consts::c*pow(tempR,3)/(kapR*densityR))/(1+pow(tauR,-2));
            Dl = (4/3*consts::a*consts::c*pow(tempL,3)/(kapL*densityL))/(1+pow(tauL,-2));
            Dt = (4/3*consts::a*consts::c*pow(tempT,3)/(kapT*densityT))/(1+pow(tauT,-2));
            Db = (4/3*consts::a*consts::c*pow(tempB,3)/(kapB*densityB))/(1+pow(tauB,-2));
            
            // D at bounds with inverse distance-weighted averages
            Dr2 = ((distR - cell->getWidth()/2) * D + cell->getWidth()/2 * Dr) / distR;
            Dl2 = ((distL - cell->getWidth()/2) * D + cell->getWidth()/2 * Dl) / distL;
            Dt2 = ((distT - cell->getHeight()/2) * D + cell->getHeight()/2 * Dt) / distT;
            Db2 = ((distB - cell->getHeight()/2) * D + cell->getHeight()/2 * Db) / distB;

            long double dEx = (Dr2*(tempR-temp)/distR - Dl2*(temp-tempL)/distL)/cell->getWidth();
            long double dEy = (Dt2*(tempT-temp)/distT - Db2*(temp-tempB)/distB)/cell->getHeight();
            long double dE = dEx + dEy;

            if (dE*getdt() > 1e15) {
                printID(getID(cell->depth, cell->index));
                std::cout << " " << opacity(density,temp,X) << " " << density << " " << densityB << " " << D << " " << Dt2 << " " << Db << " " << tauB << "\n T: " << temp << " " << tempT << " " << tempB << " dE:" << dE*getdt() << " dEx,dEy: " << dEx << "," << dEy << std::endl;
            } else {
                // std::cout << opacity(density,temp,X) << " " << density << " " << D << " " << Dt2 << " " << Db << " " << tauB << "\n T: " << temp << " " << tempT << " " << tempB << " dE:" << dE*getdt() << " dEx,dEy: " << dEx << "," << dEy << std::endl;
            }
            
            return dE;

        }

        void advect() {
            bool minSizeSmall = true; 
            long double thisMaxDensity = 0;
            maxV = 0;
            // avgFracTraveled = 0;
            // std::cout << "advect\n";
			int threadsPerBlock = 256;
			int totThreads = std::min((int) pow(4,maxDepth), (int) leafCells.size());
			int blocks = totThreads / threadsPerBlock + 1;
			
			launch_advect_kernel(blocks, threadsPerBlock, leafCells, 1, minSizeSmall, minSize, maxFracTraveled, thisMaxDensity, maxDensity, totMass, maxV, getdt());
			
			launch_advect_kernel(blocks, threadsPerBlock, leafCells, 2, minSizeSmall, minSize, maxFracTraveled, thisMaxDensity, maxDensity, totMass, maxV, getdt());

            maxDensity = thisMaxDensity;
            if (minSizeSmall) minSize *= 2;
			launch_advect_kernel(blocks, threadsPerBlock, leafCells, 3, minSizeSmall, minSize, maxFracTraveled, thisMaxDensity, maxDensity, totMass, maxV, getdt());
			
            totMass = 0;
			launch_advect_kernel(blocks, threadsPerBlock, leafCells, 4, minSizeSmall, minSize, maxFracTraveled, thisMaxDensity, maxDensity, totMass, maxV, getdt());
			
        }

        void setAMR(FluidCell *cell) {
            cell->refinedThisStep = false;
            FluidCell *cellR = cell->getRight();
            FluidCell *cellL = cell->getLeft();
            FluidCell *cellT = cell->getTop();
            FluidCell *cellB = cell->getBottom();
            cellR->absorbChildProps(); cellL->absorbChildProps(); cellT->absorbChildProps(); cellB->absorbChildProps();
            // long double mass = cell->getMass();
            // long double massR = cellR->getMass();
            // long double massL = cellL->getMass();
            // long double massT = cellT->getMass();
            // long double massB = cellB->getMass();
            long double density = cell->getDensity();
            long double densityR = cellR->getDensity();
            long double densityL = cellL->getDensity();
            long double densityT = cellT->getDensity();
            long double densityB = cellB->getDensity();

            if (abs(densityR - density)/density > refineThresh) {
                cell->shouldRefine = true;
                // cellR->shouldRefine = true;
            } else if (abs(densityR - density)/density < coarseThresh) {
                cell->shouldCoarsen = true;
            } else {
                cell->shouldCoarsen = false;
            }
            if (abs(densityL - density)/density > refineThresh) {
                cell->shouldRefine = true;
                // cellL->shouldRefine = true;
            } else if (abs(densityL - density)/density < coarseThresh) {
                cell->shouldCoarsen = true;
            } else {
                cell->shouldCoarsen = false;
            }
            if (abs(densityT - density)/density > refineThresh) {
                // std::cout << "grad " << abs(massT - mass)/mass << std::endl;
                // std::cout << massT << ", " << mass << std::endl;
                cell->shouldRefine = true;
                // cellT->shouldRefine = true;
            } else if (abs(densityT - density)/density < coarseThresh) {
                cell->shouldCoarsen = true;
            } else {
                cell->shouldCoarsen = false;
            }
            if (abs(densityB - density)/density > refineThresh) {
                cell->shouldRefine = true;
                // cellB->shouldRefine = true;
            } else if (abs(densityB - density)/density < coarseThresh) {
                cell->shouldCoarsen = true;
            } else {
                cell->shouldCoarsen = false;
            }

            FluidCell *neighbors[4] = {cellT, cellB, cellL, cellR};
            std::vector<FluidCell *> realNeighbs;
            bool island = true;
            bool willJump = false;
            for (int i = 0; i < 4; i++) {
                if (neighbors[i] != cell) {
                    realNeighbs.push_back(neighbors[i]);
                }
            }
            for (int i = 0; i < realNeighbs.size(); i++) {
                island = island & (int)realNeighbs[i]->hasChildren();
                willJump = willJump | (int)realNeighbs[i]->hasChildren();
            }
            // avoiding flickering of corners
            if (realNeighbs.size() < 4) island = false;
            if (island) {
                cell->shouldRefine = true;
                cell->shouldCoarsen = false;
            } else if (density < densCoarseThresh*maxDensity) {
                cell->shouldRefine = false;
                cell->shouldCoarsen = true;
            } else if (density > densRefineThresh*maxDensity) {
                cell->shouldRefine = true;
                cell->shouldCoarsen = false;
            }
            if (cell->shouldCoarsen) {
                if (cellR->shouldRefine || cellL->shouldRefine || cellT->shouldRefine || cellB->shouldRefine) {
                    cell->shouldCoarsen = false;
                }
                if (willJump) {
                    cell->shouldCoarsen = false;
                }
            }
            if (cell->depth > maxDepth) {
                cell->shouldCoarsen = true;
                cell->shouldRefine = false;
            }

        }

        void checkAMR() {
            std::map<uint64_t,FluidCell*> dictCopy = idToCell;
            FluidCell *cell;
            for (int i = 0; i < leafCells.size(); i++) {
                cell = leafCells[i];
                if (cell->shouldRefine) {
                    TreeLoc loc = getLocFromID(getIDfromLeaf(cell));
                    cell->shouldRefine = false;
                    cell->shouldCoarsen = false;
                    if (loc.depth < maxDepth) {
                        if (!cell->refinedThisStep) refineCell(loc.depth, loc.index);
                    }
                } else if (cell->shouldCoarsen) {
                    uint64_t id = getIDfromLeaf(cell);
                    
                    TreeLoc loc = getLocFromID(id);
                    
                    if (loc.depth > minDepth) {  
                        FluidCell *parent = cell->parent;
                        FluidCell *siblings[4] = {parent->nw, parent->ne, parent->sw, parent->se};
                        bool agreement = true;
                        for (int i = 0; i < 4; i++) {
                            agreement = siblings[i]->shouldCoarsen && agreement;

                            siblings[i]->shouldCoarsen = false;
                            siblings[i]->shouldRefine = false;
                        }
                        
                        if (agreement) {
                            Direction dirs[4] = {DOWN, LEFT, UP, RIGHT};
                            bool resJump = false;
                            for (int i = 0; i < 4; i++) {
                                FluidCell *left = siblings[i]->getLeft();
                                FluidCell *right = siblings[i]->getRight();
                                FluidCell *top = siblings[i]->getTop();
                                FluidCell *bottom = siblings[i]->getBottom();
                                FluidCell *neighbs[4] = {left,right,top,bottom};

                                for (int j = 0; j < 4; j++) {
                                    if (neighbs[j] != cell) {
                                        resJump = resJump | (neighbs[j]->hasChildren() | neighbs[j]->shouldRefine);
                                    }
                                }                         
                            }                            
                            if (!resJump) coarsenCell(loc.depth-1, loc.index >> 2);
                        }
                    }
                    cell->shouldRefine = false;
                    cell->shouldCoarsen = false;
                }
            }
            setNeighbors();
        }

        void updateLeafCells() {
            leafCells.clear();
            for (auto& pair : idToCell) {
                leafCells.push_back(pair.second);
            }
        }

        void update() {
            // double new_dt = 0.4*minSize / maxV;
            double new_dt = dt*1/maxFracTraveled;
            // double new_dt = dt*0.05/avgFracTraveled;
            dt = std::max(dt*0.1,std::min(new_dt, dt*1.05));
            T += dt;
            // dt = 100;
            // dt = std::min(new_dt, dt*1.05);
            // dt = 0.9 * dt + 0.1 * new_dt; 
            maxV = 0;
            maxFracTraveled = 0;
            avgFracTraveled = 0;
            auto start = std::chrono::steady_clock::now();
            solveGravPotential(10);
            auto grav = std::chrono::steady_clock::now();
            updateVelocities();
            auto vel = std::chrono::steady_clock::now();
            energyUpdate();
            auto nrg = std::chrono::steady_clock::now();
            advect();
            auto adv = std::chrono::steady_clock::now();
            std::cout << "\r" << totMass << " " << dt << " " << std::flush;

            for (int i = 0; i < leafCells.size(); i++) {
                tote += leafCells[i]->getE(false)*leafCells[i]->getSize();
                //if (T < 1000) {
		//setAMR(leafCells[i]);
                //}
            }
            auto setamr = std::chrono::steady_clock::now();
            checkAMR();
            auto amr = std::chrono::steady_clock::now();
            updateLeafCells();
            auto leaf = std::chrono::steady_clock::now();

            if (leafCells.size() < 0.5*pow(4,minDepth+2)) maxDepth += 1;

        }

        void freeGrid() {
            for (int i = 0; i < leafCells.size(); i++) {
                free(leafCells[i]);
            }
        }

        std::vector<FluidCell*> getLeafCells() {
            return leafCells;
        }
        

    private:
        int maxDepth = 8;
        int minDepth = 3;
        // gradient thresh
        float coarseThresh = 0.001; 
        float refineThresh = 0.1;
        // density threshold
        float densCoarseThresh = 0.0001;
        float densRefineThresh = 0.5;
        float viscosity = 0.1;
        double width, height;
        double maxV;
        double minSize; // minimum side length
        // max fraction of a cell traveled in the grid
        double maxFracTraveled, avgFracTraveled; 
        float dt;
        float T = 0;
        FluidCell *root;
        std::map<uint64_t,FluidCell*> idToCell;
        std::vector<FluidCell*> leafCells;
        long double maxDensity;
        long double totMass, tote;
};

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


void printBinaryRecursive(uint64_t num) {
    if (num > 1) {
        printBinaryRecursive(num / 2);
    }
    std::cout << (num % 2);
}


int main(int argv, char **argc) {
    if (argv > 4) std::srand((unsigned) atoi(argc[4]));
    else std::srand((unsigned) std::time(NULL));
    Simulator sim(1e9,1e9,atoi(argc[1]), atoi(argc[2]), atoi(argc[3]), "outputGrid_1e8.txt");
    
    SDL_Event event;
    if (atoi(argc[2])) sim.drawCells();

    SDL_PollEvent(&event);
    while(!(event.type == SDL_QUIT)) {
        SDL_PollEvent(&event);
        // sim.drawCells();
        // if (event.key.state == SDLK_SPACE) {
            sim.step(event);
        // }
        // std::cout << "\r" << sim.getT()/1e8 << std::endl;
        if (atoi(argc[3]) && sim.getT() > 1e8) break;
    }
    sim.freeSim();
    return 0;
}
