#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <array>
#include <map>
#include <unordered_map>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>


#include <CGAL/Point_set_2.h>
#include <CGAL/Point_set_3.h>

#include <CGAL/Surface_mesh.h>

#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Polygon_mesh_processing/measure.h>

#include <CGAL/optimal_bounding_box.h>

#include <CGAL/Real_timer.h>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_2 = CGAL::Point_2<K>;
using Point_3 = CGAL::Point_3<K>;
using Surface_mesh_3 = CGAL::Surface_mesh<Point_3>;


typedef K::Vector_3                                            Vector;
typedef CGAL::Surface_mesh<Point_3>                              Surface_mesh;
typedef boost::graph_traits<Surface_mesh>::vertex_descriptor   vertex_descriptor;

namespace py = pybind11;
namespace CP = CGAL::parameters;


PYBIND11_MODULE(pycgal, m)
{
    m.doc() = "python binding for CGAL (https://doc.cgal.org/latest/Manual/index.html)";

    py::module m_init = m.def_submodule("init", "Pybinding of initialization");

    py::class_<Point_3>(m, "Point_3")
    .def(py::init<double, double, double>(), py::arg("x"), py::arg("y"), py::arg("z"))
    .def(py::init([](const py::array_t<double> &pos){return new Point_3(pos.at(0), pos.at(1), pos.at(2));}), py::arg("pos"))
    .def_property_readonly("x", &Point_3::x)
    .def_property_readonly("y", &Point_3::y)
    .def_property_readonly("z", &Point_3::z);

    py::class_<Point_2>(m, "Point_2")
    .def(py::init<double, double>(), py::arg("x"), py::arg("y"))
    .def(py::init([](const py::array_t<double> &pos){return new Point_2(pos.at(0), pos.at(1));}), py::arg("pos"))
    .def_property_readonly("x", &Point_2::x)
    .def_property_readonly("y", &Point_2::y);

    py::module m_obb = m.def_submodule("obb", "Pybinding of optimal_bounding_box");
    m_obb.def("oriented_bounding_box", [](const py::array_t<double> &points){
        auto points_buffer = points.request();
        std::array<Point_3, 8> obb_points;
        std::vector <Point_3> point_set;

        py::array_t<double> result = py::array_t<double>(24);
        auto result_buffer = result.request();
        double *points_ptr = (double *) points_buffer.ptr, *result_ptr = (double *) result_buffer.ptr;

        for (int i = 0; i < points_buffer.shape[0]; i++)
            point_set.push_back(Point_3(points_ptr[i * 3 + 0], points_ptr[i * 3 + 1], points_ptr[i * 3 + 2]));
        CGAL::oriented_bounding_box(point_set, obb_points);

        for (int i = 0; i < 8; i++)
        {
            result_ptr[i * 3] = obb_points[i].x();
            result_ptr[i * 3 + 1] = obb_points[i].y();
            result_ptr[i * 3 + 2] = obb_points[i].z();
        }
        result.resize({8, 3});
        return result;
    }, py::arg("points"));


}

