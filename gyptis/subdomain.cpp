
#include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;
using namespace pybind11::literals;
#include <dolfin/function/Expression.h>
#include <dolfin/mesh/MeshFunction.h>

class SubdomainCpp : public dolfin::Expression
{
public:

// Create scalar expression
SubdomainCpp() : dolfin::Expression() {
}

// Function for evaluating expression on each cell
void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& cell) const override
{

        for (auto item : mapping) {
                py::str id = item.first.cast<py::str>();


                if ((*markers)[cell.index] == subdomains[id].cast<int>())
                        if (py::hasattr(item.second,"__call__"))
                                values[0] = item.second(x).cast<double>();
                        else
                                values[0] = item.second.cast<double>();
        }

}

// The data stored in mesh functions
std::shared_ptr<dolfin::MeshFunction<size_t> > markers;
py::dict subdomains;
py::dict mapping;

};

PYBIND11_MODULE(SIGNATURE, m)
{
        py::class_<SubdomainCpp, std::shared_ptr<SubdomainCpp>, dolfin::Expression>
                (m, "SubdomainCpp")
        .def(py::init<>())
        .def_readwrite("markers", &SubdomainCpp::markers)
        .def_readwrite("subdomains", &SubdomainCpp::subdomains)
        .def_readwrite("mapping", &SubdomainCpp::mapping);
}
