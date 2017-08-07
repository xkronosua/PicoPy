from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext as build_ext
import os
import numpy

include_dirs = ['./include', numpy.get_include()]
package_data = {'picopy' : ['PS5000a.dll', 'PicoIpp.dll', 'PS5000a.lib',
                'PS4000.dll', 'PS4000.lib', 'PS3000a.dll', 'PS3000a.lib',
                'USBPT104.dll', 'USBPT104.lib']}

ext_modules = [
        Extension('picopy.pico5k',
            sources=['picopy/pico5k.pyx'],
            libraries=['PS5000a'],
            include_dirs=include_dirs,
            library_dirs=['./include']),
        Extension('picopy.pico4k',
            sources=['picopy/pico4k.pyx'],
            libraries=['PS4000'],
            include_dirs=include_dirs,
            library_dirs=['./include']),
        Extension('picopy.pico3k',
            sources=['picopy/pico3k.pyx'],
            libraries=['PS3000a'],
            include_dirs=include_dirs,
            library_dirs=['./include']),
        Extension('picopy.pt104',
            sources=['picopy/pt104.pyx'],
            libraries=['usbpt104'],
            include_dirs=include_dirs,
            library_dirs=['./include']),
        Extension(
            'picopy.pico_status',
            sources = ['picopy/pico_status.pyx'], 
            include_dirs=include_dirs)]
        
setup_args = {
        'name': 'PicoPy',
        'version': '0.0.1',
        'author': 'Henry Gomersall',
        'author_email': 'heng@kedevelopments.co.uk',
        'description': 'A pythonic wrapper around the PicoScope (3000 to 5000 series) API.',
        'url': 'http://hgomersall.github.com/PicoPy/',
        'long_description': '',
        'classifiers': [
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Development Status :: 3 - Alpha',
            'License :: OSI Approved :: GNU General Public License (GPL)',
            'Operating System :: OS Independent',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering',
            ],
        'packages':['picopy'],
        'ext_modules': ext_modules,
        'include_dirs': include_dirs,
        'package_data': package_data,
        'cmdclass': {'build_ext': build_ext},
  }

if __name__ == '__main__':
    setup(**setup_args)
