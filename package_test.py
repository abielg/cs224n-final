import pkgutil

# this is the package we are inspecting -- for example 'email' from stdlib
import tensorflow

package = tensorflow
prefix = package.__name__ + "."
for importer, modname, ispkg in pkgutil.iter_modules(package.__path__, prefix):
    print "Found submodule %s (is a package: %s)" % (modname, ispkg)
    module = __import__(modname, fromlist="dummy")
    print "Imported", module
