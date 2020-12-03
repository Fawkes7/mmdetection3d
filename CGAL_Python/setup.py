from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import argparse
#from pybind11 import get_cmake_dir
from glob import glob
import sys, os, subprocess, platform, shutil


parser = argparse.ArgumentParser()
parser.add_argument('--profile', action='store_true')
parser.add_argument('--debug', action='store_true')
args, unknown = parser.parse_known_args()


def get_abs_path(path):
    ret = []
    for i in range(len(path)):
        ret.append(os.path.abspath(path[i]))
    return ret


class Pybind11Extension(Extension):
    def __init__(self, name, sources, *args, include_dirs=None, library_dirs=None, libraries=None, **kw):
        pybind_include_dirs = ['./pybind11/include']
        include_dirs = (include_dirs or []) + pybind_include_dirs
        library_dirs = library_dirs or []
        libraries = libraries or []
        kw.update(dict(include_dirs=get_abs_path(include_dirs), library_dirs=get_abs_path(library_dirs),
                       libraries=get_abs_path(libraries), language="c++",))
        super().__init__(name, get_abs_path(sources), *args, **kw)
        # print(os.path.basename(__file__))

        self.sourcedir = os.path.abspath(os.path.dirname(__file__))
        # print(self.sourcedir, os.path.dirname(__file__))
        # exit(0)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions:" +
                               ", ".join(e.name for e in self.extensions))

        assert platform.system() == "Linux"
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        original_full_path = self.get_ext_fullpath(ext.name)
        # print(ext.name, original_full_path)
        # exit(0)
        ext_dir = os.path.abspath(os.path.dirname(original_full_path))
        # ext_dir = os.path.join(ext_dir)

        cfg = 'Debug' if args.debug else 'Release'
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + ext_dir, '-DPYTHON_EXECUTABLE=' + sys.executable,
                      '-DCMAKE_BUILD_TYPE=' + cfg]
        build_args = ['--config', cfg, '--', '-j8']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''), self.distribution.get_version())

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.', "--target", "pycgal"] + build_args, cwd=self.build_temp)

        '''
        glsl_target_path = os.path.join(self.build_lib, 'sapien', 'glsl_shader')
        if os.path.exists(glsl_target_path):
            shutil.rmtree(glsl_target_path)
        shutil.copytree(os.path.join(self.build_temp, 'glsl_shader'), glsl_target_path)

        if args.optix_home:
            ptx_target_path = os.path.join(self.build_lib, 'sapien', 'ptx')
            print(ptx_target_path)
            if os.path.exists(ptx_target_path):
                shutil.rmtree(ptx_target_path)
            shutil.copytree(os.path.join(self.build_temp, 'ptx'), ptx_target_path)

        spv_path = os.path.join(self.build_lib, 'sapien', 'spv')
        spv_build_path = os.path.join(self.build_temp, 'spv')
        if os.path.exists(spv_path):
            shutil.rmtree(spv_path)
        if os.path.exists(spv_build_path):
            shutil.copytree(spv_build_path, spv_path) 
        '''

#find_package(CGAL CONFIG REQUIRED)

Pybind11Extension("pycgal", sorted(glob("src/*.cpp")))



__version__ = "1.0"

ext_modules = [
    #Pybind11Extension("pycgal",
    #    sorted(glob("src/*.cpp")),
    #    define_macros=[('VERSION_INFO', __version__)]),

    Pybind11Extension("pycgal", sorted(glob("src/*.cpp")))
]


setup(
    name="pycgal",
    version=__version__,
    author="Zhan Ling",
    description="Python wrap for cgal using pybind",
    long_description="",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
