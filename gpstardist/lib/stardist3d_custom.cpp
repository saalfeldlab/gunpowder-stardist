#include <Python.h>
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"
#include "stardist3d_custom_impl.h"

// dist.shape = (n_polys, n_rays)
// points.shape = (n_polys, 3)
// verts.shape = (n_rays, 3)
// faces.shape = (n_faces, 3)
// expects that polys are sorted with associated descending scores
// returns boolean vector of polys indices that are kept



// create the star convex distances from a labeled image
static PyObject* c_star_dist3d_max(PyObject *self, PyObject *args) {

  PyArrayObject *src = NULL;
  PyArrayObject *dst = NULL;

  PyArrayObject *pdx = NULL;
  PyArrayObject *pdy = NULL;
  PyArrayObject *pdz = NULL;


  int n_rays;
  int grid_x, grid_y, grid_z;
  float max_dist;


  if (!PyArg_ParseTuple(args, "O!O!O!O!ifiii", &PyArray_Type, &src, &PyArray_Type, &pdz ,&PyArray_Type, &pdy,
  &PyArray_Type, &pdx, &n_rays,&max_dist,&grid_z,&grid_y,&grid_x))
    return NULL;

  npy_intp *dims = PyArray_DIMS(src);

  npy_intp dims_dst[4];
  dims_dst[0] = dims[0]/grid_z;
  dims_dst[1] = dims[1]/grid_y;
  dims_dst[2] = dims[2]/grid_x;
  dims_dst[3] = n_rays;
  const float max_dist2 = max_dist * max_dist;
  dst = (PyArrayObject*)PyArray_SimpleNew(4,dims_dst,NPY_FLOAT32);

# pragma omp parallel for schedule(dynamic)
  for (int i=0; i<dims_dst[0]; i++) {
    for (int j=0; j<dims_dst[1]; j++) {
      for (int k=0; k<dims_dst[2]; k++) {
        const unsigned short value = *(unsigned short *)PyArray_GETPTR3(src,i*grid_z,j*grid_y,k*grid_x);
        // background pixel
        if (value == 0) {
          for (int n = 0; n < n_rays; n++) {
            *(float *)PyArray_GETPTR4(dst,i,j,k,n) = 0;
          }
          // foreground pixel
        } else {

          for (int n = 0; n < n_rays; n++) {

            float dx = *(float *)PyArray_GETPTR1(pdx,n);
            float dy = *(float *)PyArray_GETPTR1(pdy,n);
            float dz = *(float *)PyArray_GETPTR1(pdz,n);

            // dx /= 4;
            // dy /= 4;
            // dz /= 4;
            float x = 0, y = 0, z=0;
            // move along ray
            while (1) {
              x += dx;
              y += dy;
              z += dz;
              const int ii = round_to_int(i*grid_z+z), jj = round_to_int(j*grid_y+y), kk = round_to_int(k*grid_x+x);
              const int x2 = round_to_int(x), y2 = round_to_int(y), z2 = round_to_int(z);
              const int dist2 = x2*x2 + y2*y2 + z2*z2;
              //std::cout<<"ii: "<<ii<<" vs  "<<i*grid_z+z<<std::endl;

              // stop if out of bounds or reaching a pixel with a different value/id
              if (ii < 0 || ii >= dims[0] ||
                  jj < 0 || jj >= dims[1] ||
                  kk < 0 || kk >= dims[2] ||
                  dist2 >= max_dist2 ||
                  value != *(unsigned short *)PyArray_GETPTR3(src,ii,jj,kk))
                {
                  if (dist2 >= max_dist2) {
                    *(float *)PyArray_GETPTR4(dst,i,j,k,n) = max_dist;
                    } else {
                      const float dist = sqrt(dist2);
                      *(float *)PyArray_GETPTR4(dst,i,j,k,n) = dist;
                    }
                  // const float dist = sqrt(x*x + y*y + z*z);
                  // *(float *)PyArray_GETPTR4(dst,i,j,k,n) = dist;

                  // small correction as we overshoot the boundary
                  // const float t_corr = .5f/fmax(fmax(fabs(dx),fabs(dy)),fabs(dz));
                  // printf("%.2f\n", t_corr);
                  // x += (t_corr-1.f)*dx;
                  // y += (t_corr-1.f)*dy;
                  // z += (t_corr-1.f)*dz;
                  // const float dist = sqrt(x*x + y*y + z*z);
                  // *(float *)PyArray_GETPTR4(dst,i,j,k,n) = dist;

                  break;
                }
            }
          }
        }

      }
    }
  }

  return PyArray_Return(dst);
}


//------------------------------------------------------------------------

static struct PyMethodDef methods[] = {
                                       {"c_star_dist3d_max",
                                        c_star_dist3d_max,
                                        METH_VARARGS,
                                        "star dist 3d calculation with a max distance"},

                                       {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
                                       PyModuleDef_HEAD_INIT,
                                       "stardist3d_custom",
                                       NULL,
                                       -1,
                                       methods,
                                       NULL,NULL,NULL,NULL
};

PyMODINIT_FUNC PyInit_stardist3d_custom(void) {
  import_array();
  return PyModule_Create(&moduledef);
}
