
#include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;
using namespace pybind11::literals;
#include <dolfin/function/Expression.h>
#include <dolfin/mesh/MeshFunction.h>

class PermittivityCpp : public dolfin::Expression
{
public:

// Create scalar expression
PermittivityCpp() : dolfin::Expression() {
}

// Function for evaluating expression on each cell
void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& cell) const override
{
        // const uint cell_index = cell.index;
        // auto key = item.first.cast<std::string>()
        // auto val = item.second.cast<std::double>();
        // id = item.first.cast<str>();
        // py::print(py::str("key: {}, value={}").format(item.first, item.second));
        for (auto item : value) {
                py::str id = item.first.cast<py::str>();
                // double val = item.second.cast<double>();


                if ((*markers)[cell.index] == subdomains[id].cast<int>())
                        if (py::hasattr(item.second,"__call__"))
                                values[0] = item.second(x).cast<double>(); //item.second(x);
                        else
                                values[0] = item.second.cast<double>();
        }

}

// The data stored in mesh functions
std::shared_ptr<dolfin::MeshFunction<size_t> > markers;
py::dict subdomains;
py::dict value;

};

PYBIND11_MODULE(SIGNATURE, m)
{
        py::class_<PermittivityCpp, std::shared_ptr<PermittivityCpp>, dolfin::Expression>
                (m, "PermittivityCpp")
        .def(py::init<>())
        .def_readwrite("markers", &PermittivityCpp::markers)
        .def_readwrite("subdomains", &PermittivityCpp::subdomains)
        .def_readwrite("value", &PermittivityCpp::value);
}
