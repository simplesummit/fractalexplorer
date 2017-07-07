#!/usr/bin/env python
"""
  particle_sim.py -- a simple particle simulation, with optional CPU/GPU support

  Copyright 2016-2017 ChemicalDevelopment

  This file is part of the fractalrender project.

  FractalRender source code, as well as any other resources in this project are
free software; you are free to redistribute it and/or modify them under
the terms of the GNU General Public License; either version 3 of the
license, or any later version.

  These programs are hopefully useful and reliable, but it is understood
that these are provided WITHOUT ANY WARRANTY, or MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GPLv3 or email at
<info@chemicaldevelopment.us> for more info on this.

  Here is a copy of the GPL v3, which this software is licensed under. You
can also find a copy at http://www.gnu.org/licenses/.
"""

import argparse

parser = argparse.ArgumentParser(description='A simple particle simulation')

args = parser.parse_args()



import particle_compute


print ("doing particle_compute.main()")

particle_compute.main()

print ("done")
