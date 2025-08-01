//------------------------------------------------------------------------------
// SuiteSparse_config/SuiteSparse_config.h: common utilites for SuiteSparse
//------------------------------------------------------------------------------

// SuiteSparse_config, Copyright (c) 2012-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: BSD-3-clause

//------------------------------------------------------------------------------

// Configuration file for SuiteSparse: a Suite of Sparse matrix packages: AMD,
// COLAMD, CCOLAMD, CAMD, CHOLMOD, UMFPACK, CXSparse, SuiteSparseQR, ParU, ...

// The SuiteSparse_config.h file is configured by CMake to be specific to the
// C/C++ compiler and BLAS library being used for SuiteSparse.  The original
// file is SuiteSparse_config/SuiteSparse_config.h.in.  Do not edit the
// SuiteSparse_config.h file directly.

#ifndef SUITESPARSE_CONFIG_H
#define SUITESPARSE_CONFIG_H

//------------------------------------------------------------------------------
// SuiteSparse-wide ANSI C11 #include files
//------------------------------------------------------------------------------

#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <inttypes.h>
#include <stddef.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <ctype.h>

//------------------------------------------------------------------------------
// SuiteSparse_long is now int64_t in SuiteSparse v6.0.0 and later
//------------------------------------------------------------------------------

// The use of SuiteSparse_long is deprecated.  User applications should use
// int64_t instead.

#undef  SuiteSparse_long
#undef  SuiteSparse_long_max
#undef  SuiteSparse_long_idd
#undef  SuiteSparse_long_id

#define SuiteSparse_long int64_t
#define SuiteSparse_long_max INT64_MAX
#define SuiteSparse_long_idd PRId64
#define SuiteSparse_long_id "%" SuiteSparse_long_idd

//------------------------------------------------------------------------------
// OpenMP
//------------------------------------------------------------------------------

#if defined ( _OPENMP )

    #include <omp.h>
    #define SUITESPARSE_OPENMP_MAX_THREADS       omp_get_max_threads ( )
    #define SUITESPARSE_OPENMP_GET_NUM_THREADS   omp_get_num_threads ( )
    #define SUITESPARSE_OPENMP_GET_WTIME         omp_get_wtime ( )
    #define SUITESPARSE_OPENMP_GET_THREAD_ID     omp_get_thread_num ( )

#else

    // OpenMP not available
    #define SUITESPARSE_OPENMP_MAX_THREADS       (1)
    #define SUITESPARSE_OPENMP_GET_NUM_THREADS   (1)
    #define SUITESPARSE_OPENMP_GET_WTIME         (0)
    #define SUITESPARSE_OPENMP_GET_THREAD_ID     (0)

#endif

//------------------------------------------------------------------------------
// MATLAB/Octave
//------------------------------------------------------------------------------

//  #if defined ( MATLAB_MEX_FILE )
//  #include "mex.h"
//  #include "matrix.h"
//  #endif

//------------------------------------------------------------------------------
// string and token handling macros
//------------------------------------------------------------------------------

// SUITESPARSE_STR: convert the content of x into a string "x"
#define SUITESPARSE_XSTR(x) SUITESPARSE_STR(x)
#define SUITESPARSE_STR(x) #x

// SUITESPARSE_CAT(x,y): concatenate two tokens
#define SUITESPARSE_CAT2(x,y) x ## y
#define SUITESPARSE_CAT(x,y) SUITESPARSE_CAT2(x,y)

//------------------------------------------------------------------------------
// determine which compiler is in use
//------------------------------------------------------------------------------

#define SUITESPARSE_COMPILER_NVCC    0
#define SUITESPARSE_COMPILER_ICX     0
#define SUITESPARSE_COMPILER_ICC     0
#define SUITESPARSE_COMPILER_CLANG   0
#define SUITESPARSE_COMPILER_GCC     0
#define SUITESPARSE_COMPILER_MSC     0
#define SUITESPARSE_COMPILER_XLC     0

#if defined ( __NVCC__ )

    // NVIDIA nvcc compiler
    #undef  SUITESPARSE_COMPILER_NVCC
    #define SUITESPARSE_COMPILER_NVCC    1

    #define SUITESPARSE_COMPILER_MAJOR __CUDACC_VER_MAJOR__
    #define SUITESPARSE_COMPILER_MINOR __CUDACC_VER_MINOR__
    #define SUITESPARSE_COMPILER_SUB   __CUDACC_VER_BUILD__
    #define SUITESPARSE_COMPILER_NAME  "nvcc"

#elif defined ( __INTEL_CLANG_COMPILER )

    // Intel icx compiler, 2022.0.0 based on clang/llvm 14.0.0
    #undef  SUITESPARSE_COMPILER_ICX
    #define SUITESPARSE_COMPILER_ICX     1

    #define SUITESPARSE_COMPILER_MAJOR __INTEL_CLANG_COMPILER
    #define SUITESPARSE_COMPILER_MINOR 0
    #define SUITESPARSE_COMPILER_SUB   0
    #define SUITESPARSE_COMPILER_NAME  __VERSION__

#elif defined ( __INTEL_COMPILER )

    // Intel icc compiler: 2021.5.0 uses "gcc 7.5 mode"
    #undef  SUITESPARSE_COMPILER_ICC
    #define SUITESPARSE_COMPILER_ICC     1

    #define SUITESPARSE_COMPILER_MAJOR __INTEL_COMPILER
    #define SUITESPARSE_COMPILER_MINOR __INTEL_COMPILER_UPDATE
    #define SUITESPARSE_COMPILER_SUB   0
    #define SUITESPARSE_COMPILER_NAME  __VERSION__

#elif defined ( __clang__ )

    // clang
    #undef  SUITESPARSE_COMPILER_CLANG
    #define SUITESPARSE_COMPILER_CLANG   1

    #define SUITESPARSE_COMPILER_MAJOR __clang_major__
    #define SUITESPARSE_COMPILER_MINOR __clang_minor__
    #define SUITESPARSE_COMPILER_SUB   __clang_patchlevel__
    #define SUITESPARSE_COMPILER_NAME  "clang " __clang_version__

#elif defined ( __xlC__ )

    // xlc
    #undef  SUITESPARSE_COMPILER_XLC
    #define SUITESPARSE_COMPILER_XLC     1

    #define SUITESPARSE_COMPILER_MAJOR ( __xlC__ / 256 )
    #define SUITESPARSE_COMPILER_MINOR \
        ( __xlC__ - 256 * SUITESPARSE_COMPILER_MAJOR)
    #define SUITESPARSE_COMPILER_SUB   0
    #define SUITESPARSE_COMPILER_NAME  "IBM xlc " SUITESPARSE_XSTR (__xlC__)

#elif defined ( __GNUC__ )

    // gcc
    #undef  SUITESPARSE_COMPILER_GCC
    #define SUITESPARSE_COMPILER_GCC     1

    #define SUITESPARSE_COMPILER_MAJOR __GNUC__
    #define SUITESPARSE_COMPILER_MINOR __GNUC_MINOR__
    #define SUITESPARSE_COMPILER_SUB   __GNUC_PATCHLEVEL__
    #define SUITESPARSE_COMPILER_NAME  "GNU gcc " \
        SUITESPARSE_XSTR (__GNUC__) "." \
        SUITESPARSE_XSTR (__GNUC_MINOR__) "." \
        SUITESPARSE_XSTR (__GNUC_PATCHLEVEL__)

#elif defined ( _MSC_VER )

    // Microsoft Visual Studio (cl compiler)
    #undef  SUITESPARSE_COMPILER_MSC
    #define SUITESPARSE_COMPILER_MSC     1

    #define SUITESPARSE_COMPILER_MAJOR ( _MSC_VER / 100 )
    #define SUITESPARSE_COMPILER_MINOR \
        ( _MSC_VER - 100 * SUITESPARSE_COMPILER_MAJOR)
    #define SUITESPARSE_COMPILER_SUB   0
    #define SUITESPARSE_COMPILER_NAME \
        "Microsoft Visual Studio " SUITESPARSE_XSTR (_MSC_VER)

#else

    // other compiler
    #define SUITESPARSE_COMPILER_MAJOR 0
    #define SUITESPARSE_COMPILER_MINOR 0
    #define SUITESPARSE_COMPILER_SUB   0
    #define SUITESPARSE_COMPILER_NAME  "other C compiler"

#endif

//------------------------------------------------------------------------------
// malloc.h: required include file for Microsoft Visual Studio
//------------------------------------------------------------------------------

#if SUITESPARSE_COMPILER_MSC
    #include <malloc.h>
#endif

// this was formerly "extern", or "__declspec ..." for Windows.
#define SUITESPARSE_PUBLIC

//------------------------------------------------------------------------------
// determine the ANSI C version
//------------------------------------------------------------------------------

#ifdef __STDC_VERSION__
// ANSI C17: 201710L
// ANSI C11: 201112L
// ANSI C99: 199901L
// ANSI C95: 199409L
#define SUITESPARSE_STDC_VERSION __STDC_VERSION__
#else
// assume ANSI C90 / C89
#define SUITESPARSE_STDC_VERSION 199001L
#endif

//------------------------------------------------------------------------------
// handle the restrict keyword
//------------------------------------------------------------------------------

#if defined ( __cplusplus )

    // C++ does not have the "restrict" keyword
    #define SUITESPARSE_RESTRICT

#elif SUITESPARSE_COMPILER_MSC

    // MS Visual Studio
    #define SUITESPARSE_RESTRICT __restrict

#elif SUITESPARSE_COMPILER_NVCC

    // NVIDIA nvcc
    #define SUITESPARSE_RESTRICT __restrict__

#elif SUITESPARSE_STDC_VERSION >= 199901L

    // ANSI C99 or later
    #define SUITESPARSE_RESTRICT restrict

#else

    // ANSI C95 and earlier: no restrict keyword
    #define SUITESPARSE_RESTRICT

#endif

#ifdef __cplusplus
extern "C"
{
#endif

//==============================================================================
// SuiteSparse_config parameters and functions
//==============================================================================

// SuiteSparse-wide parameters are placed in a single static struct, defined
// locally in SuiteSparse_config.c.  It is not meant to be updated frequently
// by multiple threads.  Rather, if an application needs to modify
// SuiteSparse_config, it should do it once at the beginning of the
// application, before multiple threads are launched.

// The intent of these function pointers is that they not be used in your
// application directly, except to assign them to the desired user-provided
// functions.  Rather, you should use the SuiteSparse_malloc/calloc, etc
// wrappers defined below to access them.

// The SuiteSparse_config_*_get methods return the contents of the struct:
void *(*SuiteSparse_config_malloc_func_get (void)) (size_t);
void *(*SuiteSparse_config_calloc_func_get (void)) (size_t, size_t);
void *(*SuiteSparse_config_realloc_func_get (void)) (void *, size_t);
void (*SuiteSparse_config_free_func_get (void)) (void *);
int (*SuiteSparse_config_printf_func_get (void)) (const char *, ...);
double (*SuiteSparse_config_hypot_func_get (void)) (double, double);
int (*SuiteSparse_config_divcomplex_func_get (void)) (double, double, double, double, double *, double *);

// The SuiteSparse_config_*_set methods modify the contents of the struct:
void SuiteSparse_config_malloc_func_set (void *(*malloc_func) (size_t));
void SuiteSparse_config_calloc_func_set (void *(*calloc_func) (size_t, size_t));
void SuiteSparse_config_realloc_func_set (void *(*realloc_func) (void *, size_t));
void SuiteSparse_config_free_func_set (void (*free_func) (void *));
void SuiteSparse_config_printf_func_set (int (*printf_func) (const char *, ...));
void SuiteSparse_config_hypot_func_set (double (*hypot_func) (double, double));
void SuiteSparse_config_divcomplex_func_set (int (*divcomplex_func) (double, double, double, double, double *, double *));

// The SuiteSparse_config_*_func methods are wrappers that call the function
// pointers in the struct.  Note that there is no wrapper for the printf_func.
// See the SUITESPARSE_PRINTF macro instead.
void *SuiteSparse_config_malloc (size_t s) ;
void *SuiteSparse_config_calloc (size_t n, size_t s) ;
void *SuiteSparse_config_realloc (void *, size_t s) ;
void SuiteSparse_config_free (void *) ;
double SuiteSparse_config_hypot (double x, double y) ;
int SuiteSparse_config_divcomplex
(
    double xr, double xi, double yr, double yi, double *zr, double *zi
) ;

void SuiteSparse_start ( void ) ;   // called to start SuiteSparse

void SuiteSparse_finish ( void ) ;  // called to finish SuiteSparse

void *SuiteSparse_malloc    // pointer to allocated block of memory
(
    size_t nitems,          // number of items to malloc (>=1 is enforced)
    size_t size_of_item     // sizeof each item
) ;

void *SuiteSparse_calloc    // pointer to allocated block of memory
(
    size_t nitems,          // number of items to calloc (>=1 is enforced)
    size_t size_of_item     // sizeof each item
) ;

void *SuiteSparse_realloc   // pointer to reallocated block of memory, or
                            ///to original block if the realloc failed.
(
    size_t nitems_new,      // new number of items in the object
    size_t nitems_old,      // old number of items in the object
    size_t size_of_item,    // sizeof each item
    void *p,                // old object to reallocate
    int *ok                 // 1 if successful, 0 otherwise
) ;

void *SuiteSparse_free      // always returns NULL
(
    void *p                 // block to free
) ;

void SuiteSparse_tic    // start the timer
(
    double tic [2]      // output, contents undefined on input
) ;

double SuiteSparse_toc  // return time in seconds since last tic
(
    double tic [2]      // input: from last call to SuiteSparse_tic
) ;

double SuiteSparse_time  // returns current wall clock time in seconds
(
    void
) ;

// returns sqrt (x^2 + y^2), computed reliably
double SuiteSparse_hypot (double x, double y) ;

// complex division of c = a/b
int SuiteSparse_divcomplex
(
    double ar, double ai,       // real and imaginary parts of a
    double br, double bi,       // real and imaginary parts of b
    double *cr, double *ci      // real and imaginary parts of c
) ;

// determine which timer to use, if any
#ifndef NTIMER
    // SuiteSparse_config itself can be compiled without OpenMP,
    // but other packages can themselves use OpenMP.  In this case,
    // those packages should use omp_get_wtime() directly.  This can
    // be done via the SUITESPARSE_TIME macro, defined below:
    #define SUITESPARSE_TIMER_ENABLED
    #define SUITESPARSE_HAVE_CLOCK_GETTIME
    #define SUITESPARSE_CONFIG_TIMER omp_get_wtime
    #if defined ( SUITESPARSE_TIMER_ENABLED )
        #if defined ( _OPENMP )
            // Avoid indirection through the library if the compilation unit
            // including this header happens to use OpenMP.
            #define SUITESPARSE_TIME (omp_get_wtime ( ))
        #else
            #define SUITESPARSE_TIME (SuiteSparse_time ( ))
        #endif
    #else
        // No timer is available
        #define SUITESPARSE_TIME (0)
    #endif
#else
    // The SuiteSparse_config timer is explictly disabled;
    // use the OpenMP timer omp_get_wtime if available.
    #undef SUITESPARSE_TIMER_ENABLED
    #undef SUITESPARSE_HAVE_CLOCK_GETTIME
    #undef SUITESPARSE_CONFIG_TIMER
    #if defined ( _OPENMP )
        #define SUITESPARSE_CONFIG_TIMER omp_get_wtime
        #define SUITESPARSE_TIME (omp_get_wtime ( ))
    #else
        #define SUITESPARSE_CONFIG_TIMER none
        #define SUITESPARSE_TIME (0)
    #endif
#endif

// SuiteSparse printf macro
#define SUITESPARSE_PRINTF(params)                          \
{                                                           \
    int (*printf_func) (const char *, ...) ;                \
    printf_func = SuiteSparse_config_printf_func_get ( ) ;  \
    if (printf_func != NULL)                                \
    {                                                       \
        (void) (printf_func) params ;                       \
    }                                                       \
}

//==============================================================================
// SuiteSparse version
//==============================================================================

// SuiteSparse is not a package itself, but a collection of packages, some of
// which must be used together (UMFPACK requires AMD, CHOLMOD requires AMD,
// COLAMD, CAMD, and CCOLAMD, etc).  A version number is provided here for the
// collection itself, which is also the version number of SuiteSparse_config.

int SuiteSparse_version     // returns SUITESPARSE_VERSION
(
    // output, not defined on input.  Not used if NULL.  Returns
    // the three version codes in version [0..2]:
    // version [0] is SUITESPARSE_MAIN_VERSION
    // version [1] is SUITESPARSE_SUB_VERSION
    // version [2] is SUITESPARSE_SUBSUB_VERSION
    int version [3]
) ;

#define SUITESPARSE_HAS_VERSION_FUNCTION

#define SUITESPARSE_DATE "July 25, 2025"
#define SUITESPARSE_MAIN_VERSION    7
#define SUITESPARSE_SUB_VERSION     11
#define SUITESPARSE_SUBSUB_VERSION  0

// version format x.y
#define SUITESPARSE_VER_CODE(main,sub) ((main) * 1000 + (sub))
#define SUITESPARSE_VERSION SUITESPARSE_VER_CODE(7, 11)

// version format x.y.z
#define SUITESPARSE__VERCODE(main,sub,patch) \
    (((main)*1000ULL + (sub))*1000ULL + (patch))
#define SUITESPARSE__VERSION SUITESPARSE__VERCODE(7,11,0)

//==============================================================================
// SuiteSparse interface to the BLAS and LAPACK libraries
//==============================================================================

// Several SuiteSparse packages rely on the BLAS/LAPACK libraries (UMFPACK
// CHOLMOD, and SPQR, and likely GraphBLAS in the future).  All of these
// packages are written in C/C++, but rely on the Fortran interface to
// BLAS/LAPACK.  SuiteSparse does not use the cblas / lapacke interfaces to
// these libraries, mainly because FindBLAS.cmake does not locate them (or at
// least does not locate their respective cblas.h and lapacke.h files).  In
// addition, the original definition of these files do not include a different
// name space for 64-bit integer versions.  Finally, Intel renames cblas.h as
// mkl_cblas.h.  As a result of these many portability issues, different
// implementations of those libraries extend them in different ways.  Thus,
// SuiteSparse simply calls the Fortran functions directly.

// However, the method for how C/C++ calling Fortran depends on the compilers
// involved.  This connection is handled by the FortranCInterface.cmake module
// of CMake.

// On typical systems (Linux with the GCC compiler for example, or on the Mac
// with clang) the Fortan name "dgemm" is called by C as "dgemm_",  Other
// systems do not append the underscore.

//------------------------------------------------------------------------------
// SUITESPARSE_FORTRAN: macros created by CMake describing how C calls Fortran
//------------------------------------------------------------------------------

// SUITESPARSE_FORTAN: for Fortran routines with no "_" in their names
// SUITESPARSE__FORTAN: for Fortran routines with "_" in their names

// The decision on which of these macros to use is based on the presence of
// underscores in the original Fortran names, not the (commonly) appended
// underscore needed for C to all the corresponding Fortran routine.

// These two macros are created by the CMake module, FortranCInterface.cmake,
// which is then used by CMake to configure this file.

// The CMAKE decision can be superceded by setting -DBLAS_NO_UNDERSCORE, so
// that "dgemm" remains "dgemm" (for MS Visual Studio for example).  Setting
// -DBLAS_UNDERSCORE changes "dgemm" to "dgemm_", the common case for Mac and
// Linux.

#if defined ( BLAS_NO_UNDERSCORE )

    // no name mangling, use lower case
    #define SUITESPARSE_FORTRAN(name,NAME)  name
    #define SUITESPARSE__FORTRAN(name,NAME) name

#elif defined ( BLAS_UNDERSCORE )

    // append an underscore, use lower case
    #define SUITESPARSE_FORTRAN(name,NAME)  name ## _
    #define SUITESPARSE__FORTRAN(name,NAME) name ## _

#else

    // let CMake decide how C calls Fortran
    #define SUITESPARSE_FORTRAN(name,NAME) name##_
    #define SUITESPARSE__FORTRAN(name,NAME) name##_

#endif

//------------------------------------------------------------------------------
// SUITESPARSE_BLAS_INT: the BLAS/LAPACK integer (int32_t or int64_t)
//------------------------------------------------------------------------------

// CMake 3.22 and later allow the selection of the BLAS/LAPACK integer size.
// This information is then used to configure this file with the definition of
// this integer: int32_t or int64_t.

// When compiling SuiteSparse for a MATLAB mexFunction, the MATLAB libmwblas is
// used, which is a 64-bit integer version of the BLAS.  CMake is not used to
// configure SuiteSparse in this case.  The flag -DBLAS64 can be used to ensure
// a 64-bit BLAS is used.  Likewise, -DBLAS32 ensures a 32-bit BLAS is used.

#if defined ( BLAS64 )

    // override the BLAS found by CMake, and force a 64-bit interface
    #define SUITESPARSE_BLAS_INT int64_t

#elif defined ( BLAS32 )

    // override the BLAS found by CMake, and force a 32-bit interface
    #define SUITESPARSE_BLAS_INT int32_t

#else

    // let CMake determine the size of the integer in the Fortran BLAS
    #define SUITESPARSE_BLAS_INT int32_t

#endif

// SUITESPARSE_TO_BLAS_INT: convert an integer k to a BLAS integer K and set ok
// to false if the conversion changes its value.  This is implemented as a
// macro so that can work with any type of the integer k.
#define SUITESPARSE_TO_BLAS_INT(K,k,ok)         \
    SUITESPARSE_BLAS_INT K = (k) ;              \
    ok = ok && ((sizeof (K) >= sizeof (k)) || ((int64_t)(K) == (int64_t)(k))) ;

//------------------------------------------------------------------------------
// SUITESPARSE_BLAS_SUFFIX: modify the name of a Fortran BLAS/LAPACK routine
//------------------------------------------------------------------------------

// OpenBLAS can be compiled by appending a suffix to each routine, so that the
// Fortan routine dgemm becomes dgemm_64, which denotes a version of dgemm with
// 64-bit integer parameters.  The Sun Performance library does the same thing,
// but without the internal underscore, as dgemm64.

// If the suffix does not contain "_", use (Sun Perf., for example):

//     cd build && cmake -DBLAS64_SUFFIX="64" ..

// If the suffix contains "_" (OpenBLAS in spack for example), use the
// following:

//     cd build && cmake -DBLAS64_SUFFIX="_64" ..

// This setting could be used by the spack packaging of SuiteSparse when linked
// with the spack-installed OpenBLAS with 64-bit integers.  See
// https://github.com/spack/spack/blob/develop/var/spack/repos/builtin/packages/suite-sparse/package.py

#if defined ( BLAS64__SUFFIX )

    // The suffix includes an undersore (such as "_64"), so the Fortran name
    // must be processed with the SUITESPARSE__FORTRAN macro.
    #define SUITESPARSE_G(name,NAME) SUITESPARSE__FORTRAN(name,NAME)
    #define SUITESPARSE_F(name,NAME)                            \
        SUITESPARSE_G (SUITESPARSE_CAT (name, BLAS64__SUFFIX),  \
                       SUITESPARSE_CAT (NAME, BLAS64__SUFFIX))
    #define SUITESPARSE_BLAS(name,NAME) SUITESPARSE_F(name,NAME)

#elif defined ( BLAS64_SUFFIX )

    // The suffix does not include an undersore, and neither do the original
    // names of the BLAS and LAPACK routines.  Thus, the Fortran name must be
    // processed with the SUITESPARSE_FORTRAN macro.
    #define SUITESPARSE_G(name,NAME) SUITESPARSE_FORTRAN(name,NAME)
    #define SUITESPARSE_F(name,NAME)                            \
        SUITESPARSE_G (SUITESPARSE_CAT (name, BLAS64_SUFFIX),  \
                       SUITESPARSE_CAT (NAME, BLAS64_SUFFIX))
    #define SUITESPARSE_BLAS(name,NAME) SUITESPARSE_F(name,NAME)

#else

    // No suffix is need, so the final Fortran name includes no suffix.
    #define SUITESPARSE_BLAS(name,NAME) SUITESPARSE_FORTRAN(name,NAME)

#endif

//------------------------------------------------------------------------------
// C names of Fortan BLAS and LAPACK functions used by SuiteSparse
//------------------------------------------------------------------------------

// double
#define SUITESPARSE_BLAS_DTRSV      SUITESPARSE_BLAS ( dtrsv  , DTRSV  )
#define SUITESPARSE_BLAS_DGEMV      SUITESPARSE_BLAS ( dgemv  , DGEMV  )
#define SUITESPARSE_BLAS_DTRSM      SUITESPARSE_BLAS ( dtrsm  , DTRSM  )
#define SUITESPARSE_BLAS_DGEMM      SUITESPARSE_BLAS ( dgemm  , DGEMM  )
#define SUITESPARSE_BLAS_DSYRK      SUITESPARSE_BLAS ( dsyrk  , DSYRK  )
#define SUITESPARSE_BLAS_DGER       SUITESPARSE_BLAS ( dger   , DGER   )
#define SUITESPARSE_BLAS_DSCAL      SUITESPARSE_BLAS ( dscal  , DSCAL  )
#define SUITESPARSE_BLAS_DNRM2      SUITESPARSE_BLAS ( dnrm2  , DNRM2  )

#define SUITESPARSE_LAPACK_DPOTRF   SUITESPARSE_BLAS ( dpotrf , DPOTRF )
#define SUITESPARSE_LAPACK_DLARF    SUITESPARSE_BLAS ( dlarf  , DLARF  )
#define SUITESPARSE_LAPACK_DLARFG   SUITESPARSE_BLAS ( dlarfg , DLARFG )
#define SUITESPARSE_LAPACK_DLARFT   SUITESPARSE_BLAS ( dlarft , DLARFT )
#define SUITESPARSE_LAPACK_DLARFB   SUITESPARSE_BLAS ( dlarfb , DLARFB )

// double complex
#define SUITESPARSE_BLAS_ZTRSV      SUITESPARSE_BLAS ( ztrsv  , ZTRSV  )
#define SUITESPARSE_BLAS_ZGEMV      SUITESPARSE_BLAS ( zgemv  , ZGEMV  )
#define SUITESPARSE_BLAS_ZTRSM      SUITESPARSE_BLAS ( ztrsm  , ZTRSM  )
#define SUITESPARSE_BLAS_ZGEMM      SUITESPARSE_BLAS ( zgemm  , ZGEMM  )
#define SUITESPARSE_BLAS_ZHERK      SUITESPARSE_BLAS ( zherk  , ZHERK  )
#define SUITESPARSE_BLAS_ZGERU      SUITESPARSE_BLAS ( zgeru  , ZGERU  )
#define SUITESPARSE_BLAS_ZSCAL      SUITESPARSE_BLAS ( zscal  , ZSCAL  )
#define SUITESPARSE_BLAS_DZNRM2     SUITESPARSE_BLAS ( dznrm2 , DZNRM2 )

#define SUITESPARSE_LAPACK_ZPOTRF   SUITESPARSE_BLAS ( zpotrf , ZPOTRF )
#define SUITESPARSE_LAPACK_ZLARF    SUITESPARSE_BLAS ( zlarf  , ZLARF  )
#define SUITESPARSE_LAPACK_ZLARFG   SUITESPARSE_BLAS ( zlarfg , ZLARFG )
#define SUITESPARSE_LAPACK_ZLARFT   SUITESPARSE_BLAS ( zlarft , ZLARFT )
#define SUITESPARSE_LAPACK_ZLARFB   SUITESPARSE_BLAS ( zlarfb , ZLARFB )

// single
#define SUITESPARSE_BLAS_STRSV      SUITESPARSE_BLAS ( strsv  , STRSV  )
#define SUITESPARSE_BLAS_SGEMV      SUITESPARSE_BLAS ( sgemv  , SGEMV  )
#define SUITESPARSE_BLAS_STRSM      SUITESPARSE_BLAS ( strsm  , STRSM  )
#define SUITESPARSE_BLAS_SGEMM      SUITESPARSE_BLAS ( sgemm  , SGEMM  )
#define SUITESPARSE_BLAS_SSYRK      SUITESPARSE_BLAS ( ssyrk  , SSYRK  )
#define SUITESPARSE_BLAS_SGER       SUITESPARSE_BLAS ( sger   , SGER   )
#define SUITESPARSE_BLAS_SSCAL      SUITESPARSE_BLAS ( sscal  , SSCAL  )
#define SUITESPARSE_BLAS_SNRM2      SUITESPARSE_BLAS ( snrm2  , SNRM2  )

#define SUITESPARSE_LAPACK_SPOTRF   SUITESPARSE_BLAS ( spotrf , SPOTRF )
#define SUITESPARSE_LAPACK_SLARF    SUITESPARSE_BLAS ( slarf  , SLARF  )
#define SUITESPARSE_LAPACK_SLARFG   SUITESPARSE_BLAS ( slarfg , SLARFG )
#define SUITESPARSE_LAPACK_SLARFT   SUITESPARSE_BLAS ( slarft , SLARFT )
#define SUITESPARSE_LAPACK_SLARFB   SUITESPARSE_BLAS ( slarfb , SLARFB )

// single complex
#define SUITESPARSE_BLAS_CTRSV      SUITESPARSE_BLAS ( ctrsv  , CTRSV  )
#define SUITESPARSE_BLAS_CGEMV      SUITESPARSE_BLAS ( cgemv  , CGEMV  )
#define SUITESPARSE_BLAS_CTRSM      SUITESPARSE_BLAS ( ctrsm  , CTRSM  )
#define SUITESPARSE_BLAS_CGEMM      SUITESPARSE_BLAS ( cgemm  , CGEMM  )
#define SUITESPARSE_BLAS_CHERK      SUITESPARSE_BLAS ( cherk  , CHERK  )
#define SUITESPARSE_BLAS_CGERU      SUITESPARSE_BLAS ( cgeru  , CGERU  )
#define SUITESPARSE_BLAS_CSCAL      SUITESPARSE_BLAS ( cscal  , CSCAL  )
#define SUITESPARSE_BLAS_SCNRM2     SUITESPARSE_BLAS ( scnrm2 , SCNRM2 )

#define SUITESPARSE_LAPACK_CPOTRF   SUITESPARSE_BLAS ( cpotrf , CPOTRF )
#define SUITESPARSE_LAPACK_CLARF    SUITESPARSE_BLAS ( clarf  , CLARF  )
#define SUITESPARSE_LAPACK_CLARFG   SUITESPARSE_BLAS ( clarfg , CLARFG )
#define SUITESPARSE_LAPACK_CLARFT   SUITESPARSE_BLAS ( clarft , CLARFT )
#define SUITESPARSE_LAPACK_CLARFB   SUITESPARSE_BLAS ( clarfb , CLARFB )

//------------------------------------------------------------------------------
// prototypes and macros for BLAS and SUITESPARSE_LAPACK functions
//------------------------------------------------------------------------------

// For complex functions, the (void *) parameters are actually pointers to
// arrays of complex values.  They are prototyped here as (void *) to allow
// them to be called from both C and C++.

// See https://netlib.org/blas/ and https://netlib.org/lapack/ for the
// definitions of the inputs/outputs of these functions.

// These prototypes need to be found by UMFPACK, CHOLMOD, and SPQR, and to do
// so, they need to appear in this public header to ensure the correct BLAS
// library and integer size is used.  However, these definitions should not
// (normally) be exposed to the user application.

// If a user application wishes to use these definitions, simply add the
// following prior to #include'ing any SuiteSparse headers (amd.h, and so on):
//
//      #define SUITESPARSE_BLAS_DEFINITIONS
//      #include "SuiteSparse_config.h"

#if defined ( SUITESPARSE_BLAS_DEFINITIONS )
#ifndef SUITESPARSE_BLAS_PROTOTYPES
#define SUITESPARSE_BLAS_PROTOTYPES
#endif
#ifndef SUITESPARSE_BLAS_MACROS
#define SUITESPARSE_BLAS_MACROS
#endif
#endif

//------------------------------------------------------------------------------
// prototypes of BLAS and SUITESPARSE_LAPACK functions
//------------------------------------------------------------------------------

#if defined ( SUITESPARSE_BLAS_PROTOTYPES )

//------------------------------------------------------------------------------
// gemv: Y = alpha*A*x + beta*Y
//------------------------------------------------------------------------------

void SUITESPARSE_BLAS_DGEMV
(
    // input:
    const char *trans,
    const SUITESPARSE_BLAS_INT *m,
    const SUITESPARSE_BLAS_INT *n,
    const double *alpha,
    const double *A,
    const SUITESPARSE_BLAS_INT *lda,
    const double *X,
    const SUITESPARSE_BLAS_INT *incx,
    const double *beta,
    // input/output:
    double *Y,
    // input:
    const SUITESPARSE_BLAS_INT *incy
) ;

void SUITESPARSE_BLAS_SGEMV
(
    // input:
    const char *trans,
    const SUITESPARSE_BLAS_INT *m,
    const SUITESPARSE_BLAS_INT *n,
    const float *alpha,
    const float *A,
    const SUITESPARSE_BLAS_INT *lda,
    const float *X,
    const SUITESPARSE_BLAS_INT *incx,
    const float *beta,
    // input/output:
    float *Y,
    // input:
    const SUITESPARSE_BLAS_INT *incy
) ;

void SUITESPARSE_BLAS_ZGEMV
(
    // input:
    const char *trans,
    const SUITESPARSE_BLAS_INT *m,
    const SUITESPARSE_BLAS_INT *n,
    const void *alpha,
    const void *A,
    const SUITESPARSE_BLAS_INT *lda,
    const void *X,
    const SUITESPARSE_BLAS_INT *incx,
    const void *beta,
    // input/output:
    void *Y,
    // input:
    const SUITESPARSE_BLAS_INT *incy
) ;

void SUITESPARSE_BLAS_CGEMV
(
    // input:
    const char *trans,
    const SUITESPARSE_BLAS_INT *m,
    const SUITESPARSE_BLAS_INT *n,
    const void *alpha,
    const void *A,
    const SUITESPARSE_BLAS_INT *lda,
    const void *X,
    const SUITESPARSE_BLAS_INT *incx,
    const void *beta,
    // input/output:
    void *Y,
    // input:
    const SUITESPARSE_BLAS_INT *incy
) ;

//------------------------------------------------------------------------------
// trsv: solve Lx=b, Ux=b, L'x=b, or U'x=b
//------------------------------------------------------------------------------

void SUITESPARSE_BLAS_DTRSV
(
    // input:
    const char *uplo,
    const char *trans,
    const char *diag,
    const SUITESPARSE_BLAS_INT *n,
    const double *A,
    const SUITESPARSE_BLAS_INT *lda,
    // input/output:
    double *X,
    // input:
    const SUITESPARSE_BLAS_INT *incx
) ;

void SUITESPARSE_BLAS_STRSV
(
    // input:
    const char *uplo,
    const char *trans,
    const char *diag,
    const SUITESPARSE_BLAS_INT *n,
    const float *A,
    const SUITESPARSE_BLAS_INT *lda,
    // input/output:
    float *X,
    // input:
    const SUITESPARSE_BLAS_INT *incx
) ;

void SUITESPARSE_BLAS_ZTRSV
(
    // input:
    const char *uplo,
    const char *trans,
    const char *diag,
    const SUITESPARSE_BLAS_INT *n,
    const void *A,
    const SUITESPARSE_BLAS_INT *lda,
    // input/output:
    void *X,
    // input:
    const SUITESPARSE_BLAS_INT *incx
) ;

void SUITESPARSE_BLAS_CTRSV
(
    // input:
    const char *uplo,
    const char *trans,
    const char *diag,
    const SUITESPARSE_BLAS_INT *n,
    const void *A,
    const SUITESPARSE_BLAS_INT *lda,
    // input/output:
    void *X,
    // input:
    const SUITESPARSE_BLAS_INT *incx
) ;

//------------------------------------------------------------------------------
// trsm: solve LX=B, UX=B, L'X=B, or U'X=B
//------------------------------------------------------------------------------

void SUITESPARSE_BLAS_DTRSM
(
    // input:
    const char *side,
    const char *uplo,
    const char *transa,
    const char *diag,
    const SUITESPARSE_BLAS_INT *m,
    const SUITESPARSE_BLAS_INT *n,
    const double *alpha,
    const double *A,
    const SUITESPARSE_BLAS_INT *lda,
    // input/output:
    double *B,
    // input:
    const SUITESPARSE_BLAS_INT *ldb
) ;

void SUITESPARSE_BLAS_STRSM
(
    // input:
    const char *side,
    const char *uplo,
    const char *transa,
    const char *diag,
    const SUITESPARSE_BLAS_INT *m,
    const SUITESPARSE_BLAS_INT *n,
    const float *alpha,
    const float *A,
    const SUITESPARSE_BLAS_INT *lda,
    // input/output:
    float *B,
    // input:
    const SUITESPARSE_BLAS_INT *ldb
) ;

void SUITESPARSE_BLAS_ZTRSM
(
    // input:
    const char *side,
    const char *uplo,
    const char *transa,
    const char *diag,
    const SUITESPARSE_BLAS_INT *m,
    const SUITESPARSE_BLAS_INT *n,
    const void *alpha,
    const void *A,
    const SUITESPARSE_BLAS_INT *lda,
    // input/output:
    void *B,
    // input:
    const SUITESPARSE_BLAS_INT *ldb
) ;

void SUITESPARSE_BLAS_CTRSM
(
    // input:
    const char *side,
    const char *uplo,
    const char *transa,
    const char *diag,
    const SUITESPARSE_BLAS_INT *m,
    const SUITESPARSE_BLAS_INT *n,
    const void *alpha,
    const void *A,
    const SUITESPARSE_BLAS_INT *lda,
    // input/output:
    void *B,
    // input:
    const SUITESPARSE_BLAS_INT *ldb
) ;

//------------------------------------------------------------------------------
// gemm: C = alpha*A*B + beta*C
//------------------------------------------------------------------------------

void SUITESPARSE_BLAS_DGEMM
(
    // input:
    const char *transa,
    const char *transb,
    const SUITESPARSE_BLAS_INT *m,
    const SUITESPARSE_BLAS_INT *n,
    const SUITESPARSE_BLAS_INT *k,
    const double *alpha,
    const double *A,
    const SUITESPARSE_BLAS_INT *lda,
    const double *B,
    const SUITESPARSE_BLAS_INT *ldb,
    const double *beta,
    // input/output:
    double *C,
    // input:
    const SUITESPARSE_BLAS_INT *ldc
) ;

void SUITESPARSE_BLAS_SGEMM
(
    // input:
    const char *transa,
    const char *transb,
    const SUITESPARSE_BLAS_INT *m,
    const SUITESPARSE_BLAS_INT *n,
    const SUITESPARSE_BLAS_INT *k,
    const float *alpha,
    const float *A,
    const SUITESPARSE_BLAS_INT *lda,
    const float *B,
    const SUITESPARSE_BLAS_INT *ldb,
    const float *beta,
    // input/output:
    float *C,
    // input:
    const SUITESPARSE_BLAS_INT *ldc
) ;

void SUITESPARSE_BLAS_ZGEMM
(
    // input:
    const char *transa,
    const char *transb,
    const SUITESPARSE_BLAS_INT *m,
    const SUITESPARSE_BLAS_INT *n,
    const SUITESPARSE_BLAS_INT *k,
    const void *alpha,
    const void *A,
    const SUITESPARSE_BLAS_INT *lda,
    const void *B,
    const SUITESPARSE_BLAS_INT *ldb,
    const void *beta,
    // input/output:
    void *C,
    // input:
    const SUITESPARSE_BLAS_INT *ldc
) ;

void SUITESPARSE_BLAS_CGEMM
(
    // input:
    const char *transa,
    const char *transb,
    const SUITESPARSE_BLAS_INT *m,
    const SUITESPARSE_BLAS_INT *n,
    const SUITESPARSE_BLAS_INT *k,
    const void *alpha,
    const void *A,
    const SUITESPARSE_BLAS_INT *lda,
    const void *B,
    const SUITESPARSE_BLAS_INT *ldb,
    const void *beta,
    // input/output:
    void *C,
    // input:
    const SUITESPARSE_BLAS_INT *ldc
) ;

//------------------------------------------------------------------------------
// syrk/herk: C = alpha*A*A' + beta*C ; or C = alpha*A'*A + beta*C
//------------------------------------------------------------------------------

void SUITESPARSE_BLAS_DSYRK
(
    // input:
    const char *uplo,
    const char *trans,
    const SUITESPARSE_BLAS_INT *n,
    const SUITESPARSE_BLAS_INT *k,
    const double *alpha,
    const double *A,
    const SUITESPARSE_BLAS_INT *lda,
    const double *beta,
    // input/output:
    double *C,
    // input:
    const SUITESPARSE_BLAS_INT *ldc
) ;

void SUITESPARSE_BLAS_SSYRK
(
    // input:
    const char *uplo,
    const char *trans,
    const SUITESPARSE_BLAS_INT *n,
    const SUITESPARSE_BLAS_INT *k,
    const float *alpha,
    const float *A,
    const SUITESPARSE_BLAS_INT *lda,
    const float *beta,
    // input/output:
    float *C,
    // input:
    const SUITESPARSE_BLAS_INT *ldc
) ;

void SUITESPARSE_BLAS_ZHERK
(
    // input:
    const char *uplo,
    const char *trans,
    const SUITESPARSE_BLAS_INT *n,
    const SUITESPARSE_BLAS_INT *k,
    const void *alpha,
    const void *A,
    const SUITESPARSE_BLAS_INT *lda,
    const void *beta,
    // input/output:
    void *C,
    // input:
    const SUITESPARSE_BLAS_INT *ldc
) ;

void SUITESPARSE_BLAS_CHERK
(
    // input:
    const char *uplo,
    const char *trans,
    const SUITESPARSE_BLAS_INT *n,
    const SUITESPARSE_BLAS_INT *k,
    const void *alpha,
    const void *A,
    const SUITESPARSE_BLAS_INT *lda,
    const void *beta,
    // input/output:
    void *C,
    // input:
    const SUITESPARSE_BLAS_INT *ldc
) ;

//------------------------------------------------------------------------------
// potrf: Cholesky factorization
//------------------------------------------------------------------------------

void SUITESPARSE_LAPACK_DPOTRF
(
    // input:
    const char *uplo,
    const SUITESPARSE_BLAS_INT *n,
    // input/output:
    double *A,
    // input:
    const SUITESPARSE_BLAS_INT *lda,
    // output:
    SUITESPARSE_BLAS_INT *info
) ;

void SUITESPARSE_LAPACK_SPOTRF
(
    // input:
    const char *uplo,
    const SUITESPARSE_BLAS_INT *n,
    // input/output:
    float *A,
    // input:
    const SUITESPARSE_BLAS_INT *lda,
    // output:
    SUITESPARSE_BLAS_INT *info
) ;

void SUITESPARSE_LAPACK_ZPOTRF
(
    // input:
    const char *uplo,
    const SUITESPARSE_BLAS_INT *n,
    // input/output:
    void *A,
    // input:
    const SUITESPARSE_BLAS_INT *lda,
    // output:
    SUITESPARSE_BLAS_INT *info
) ;

void SUITESPARSE_LAPACK_CPOTRF
(
    // input:
    const char *uplo,
    const SUITESPARSE_BLAS_INT *n,
    // input/output:
    void *A,
    // input:
    const SUITESPARSE_BLAS_INT *lda,
    // output:
    SUITESPARSE_BLAS_INT *info
) ;

//------------------------------------------------------------------------------
// scal: Y = alpha*Y
//------------------------------------------------------------------------------

void SUITESPARSE_BLAS_DSCAL
(
    // input:
    const SUITESPARSE_BLAS_INT *n,
    const double *alpha,
    // input/output:
    double *Y,
    // input:
    const SUITESPARSE_BLAS_INT *incy
) ;

void SUITESPARSE_BLAS_SSCAL
(
    // input:
    const SUITESPARSE_BLAS_INT *n,
    const float *alpha,
    // input/output:
    float *Y,
    // input:
    const SUITESPARSE_BLAS_INT *incy
) ;

void SUITESPARSE_BLAS_ZSCAL
(
    // input:
    const SUITESPARSE_BLAS_INT *n,
    const void *alpha,
    // input/output:
    void *Y,
    // input:
    const SUITESPARSE_BLAS_INT *incy
) ;

void SUITESPARSE_BLAS_CSCAL
(
    // input:
    const SUITESPARSE_BLAS_INT *n,
    const void *alpha,
    // input/output:
    void *Y,
    // input:
    const SUITESPARSE_BLAS_INT *incy
) ;

//------------------------------------------------------------------------------
// ger/geru: A = alpha*x*y' + A
//------------------------------------------------------------------------------

void SUITESPARSE_BLAS_DGER
(
    // input:
    const SUITESPARSE_BLAS_INT *m,
    const SUITESPARSE_BLAS_INT *n,
    const double *alpha,
    const double *X,
    const SUITESPARSE_BLAS_INT *incx,
    const double *Y,
    const SUITESPARSE_BLAS_INT *incy,
    // input/output:
    double *A,
    // input:
    const SUITESPARSE_BLAS_INT *lda
) ;

void SUITESPARSE_BLAS_SGER
(
    // input:
    const SUITESPARSE_BLAS_INT *m,
    const SUITESPARSE_BLAS_INT *n,
    const float *alpha,
    const float *X,
    const SUITESPARSE_BLAS_INT *incx,
    const float *Y,
    const SUITESPARSE_BLAS_INT *incy,
    // input/output:
    float *A,
    // input:
    const SUITESPARSE_BLAS_INT *lda
) ;

void SUITESPARSE_BLAS_ZGERU
(
    // input:
    const SUITESPARSE_BLAS_INT *m,
    const SUITESPARSE_BLAS_INT *n,
    const void *alpha,
    const void *X,
    const SUITESPARSE_BLAS_INT *incx,
    const void *Y,
    const SUITESPARSE_BLAS_INT *incy,
    // input/output:
    void *A,
    // input:
    const SUITESPARSE_BLAS_INT *lda
) ;

void SUITESPARSE_BLAS_CGERU
(
    // input:
    const SUITESPARSE_BLAS_INT *m,
    const SUITESPARSE_BLAS_INT *n,
    const void *alpha,
    const void *X,
    const SUITESPARSE_BLAS_INT *incx,
    const void *Y,
    const SUITESPARSE_BLAS_INT *incy,
    // input/output:
    void *A,
    // input:
    const SUITESPARSE_BLAS_INT *lda
) ;

//------------------------------------------------------------------------------
// larft: T = block Householder factor
//------------------------------------------------------------------------------

void SUITESPARSE_LAPACK_DLARFT
(
    // input:
    const char *direct,
    const char *storev,
    const SUITESPARSE_BLAS_INT *n,
    const SUITESPARSE_BLAS_INT *k,
    const double *V,
    const SUITESPARSE_BLAS_INT *ldv,
    const double *Tau,
    // output:
    double *T,
    // input:
    const SUITESPARSE_BLAS_INT *ldt
) ;

void SUITESPARSE_LAPACK_SLARFT
(
    // input:
    const char *direct,
    const char *storev,
    const SUITESPARSE_BLAS_INT *n,
    const SUITESPARSE_BLAS_INT *k,
    const float *V,
    const SUITESPARSE_BLAS_INT *ldv,
    const float *Tau,
    // output:
    float *T,
    // input:
    const SUITESPARSE_BLAS_INT *ldt
) ;

void SUITESPARSE_LAPACK_ZLARFT
(
    // input:
    const char *direct,
    const char *storev,
    const SUITESPARSE_BLAS_INT *n,
    const SUITESPARSE_BLAS_INT *k,
    const void *V,
    const SUITESPARSE_BLAS_INT *ldv,
    const void *Tau,
    // output:
    void *T,
    // input:
    const SUITESPARSE_BLAS_INT *ldt
) ;

void SUITESPARSE_LAPACK_CLARFT
(
    // input:
    const char *direct,
    const char *storev,
    const SUITESPARSE_BLAS_INT *n,
    const SUITESPARSE_BLAS_INT *k,
    const void *V,
    const SUITESPARSE_BLAS_INT *ldv,
    const void *Tau,
    // output:
    void *T,
    // input:
    const SUITESPARSE_BLAS_INT *ldt
) ;

//------------------------------------------------------------------------------
// larfb: apply block Householder reflector
//------------------------------------------------------------------------------

void SUITESPARSE_LAPACK_DLARFB
(
    // input:
    const char *side,
    const char *trans,
    const char *direct,
    const char *storev,
    const SUITESPARSE_BLAS_INT *m,
    const SUITESPARSE_BLAS_INT *n,
    const SUITESPARSE_BLAS_INT *k,
    const double *V,
    const SUITESPARSE_BLAS_INT *ldv,
    const double *T,
    const SUITESPARSE_BLAS_INT *ldt,
    // input/output:
    double *C,
    // input:
    const SUITESPARSE_BLAS_INT *ldc,
    // workspace:
    double *Work,
    // input:
    const SUITESPARSE_BLAS_INT *ldwork
) ;

void SUITESPARSE_LAPACK_SLARFB
(
    // input:
    const char *side,
    const char *trans,
    const char *direct,
    const char *storev,
    const SUITESPARSE_BLAS_INT *m,
    const SUITESPARSE_BLAS_INT *n,
    const SUITESPARSE_BLAS_INT *k,
    const float *V,
    const SUITESPARSE_BLAS_INT *ldv,
    const float *T,
    const SUITESPARSE_BLAS_INT *ldt,
    // input/output:
    float *C,
    // input:
    const SUITESPARSE_BLAS_INT *ldc,
    // workspace:
    float *Work,
    // input:
    const SUITESPARSE_BLAS_INT *ldwork
) ;

void SUITESPARSE_LAPACK_ZLARFB
(
    // input:
    const char *side,
    const char *trans,
    const char *direct,
    const char *storev,
    const SUITESPARSE_BLAS_INT *m,
    const SUITESPARSE_BLAS_INT *n,
    const SUITESPARSE_BLAS_INT *k,
    const void *V,
    const SUITESPARSE_BLAS_INT *ldv,
    const void *T,
    const SUITESPARSE_BLAS_INT *ldt,
    // input/output:
    void *C,
    // input:
    const SUITESPARSE_BLAS_INT *ldc,
    // workspace:
    void *Work,
    // input:
    const SUITESPARSE_BLAS_INT *ldwork
) ;

void SUITESPARSE_LAPACK_CLARFB
(
    // input:
    const char *side,
    const char *trans,
    const char *direct,
    const char *storev,
    const SUITESPARSE_BLAS_INT *m,
    const SUITESPARSE_BLAS_INT *n,
    const SUITESPARSE_BLAS_INT *k,
    const void *V,
    const SUITESPARSE_BLAS_INT *ldv,
    const void *T,
    const SUITESPARSE_BLAS_INT *ldt,
    // input/output:
    void *C,
    // input:
    const SUITESPARSE_BLAS_INT *ldc,
    // workspace:
    void *Work,
    // input:
    const SUITESPARSE_BLAS_INT *ldwork
) ;

//------------------------------------------------------------------------------
// nrm2: vector 2-norm
//------------------------------------------------------------------------------

double SUITESPARSE_BLAS_DNRM2
(
    // input:
    const SUITESPARSE_BLAS_INT *n,
    const double *X,
    const SUITESPARSE_BLAS_INT *incx
) ;

float SUITESPARSE_BLAS_SNRM2
(
    // input:
    const SUITESPARSE_BLAS_INT *n,
    const float *X,
    const SUITESPARSE_BLAS_INT *incx
) ;

double SUITESPARSE_BLAS_DZNRM2
(
    // input:
    const SUITESPARSE_BLAS_INT *n,
    const void *X,
    const SUITESPARSE_BLAS_INT *incx
) ;

float SUITESPARSE_BLAS_SCNRM2
(
    // input:
    const SUITESPARSE_BLAS_INT *n,
    const void *X,
    const SUITESPARSE_BLAS_INT *incx
) ;

//------------------------------------------------------------------------------
// larfg: generate Householder reflector
//------------------------------------------------------------------------------

void SUITESPARSE_LAPACK_DLARFG
(
    // input:
    const SUITESPARSE_BLAS_INT *n,
    // input/output:
    double *alpha,
    double *X,
    // input:
    const SUITESPARSE_BLAS_INT *incx,
    // output:
    double *tau
) ;

void SUITESPARSE_LAPACK_SLARFG
(
    // input:
    const SUITESPARSE_BLAS_INT *n,
    // input/output:
    float *alpha,
    float *X,
    // input:
    const SUITESPARSE_BLAS_INT *incx,
    // output:
    float *tau
) ;

void SUITESPARSE_LAPACK_ZLARFG
(
    // input:
    const SUITESPARSE_BLAS_INT *n,
    // input/output:
    void *alpha,
    void *X,
    // input:
    const SUITESPARSE_BLAS_INT *incx,
    // output:
    void *tau
) ;

void SUITESPARSE_LAPACK_CLARFG
(
    // input:
    const SUITESPARSE_BLAS_INT *n,
    // input/output:
    void *alpha,
    void *X,
    // input:
    const SUITESPARSE_BLAS_INT *incx,
    // output:
    void *tau
) ;

//------------------------------------------------------------------------------
// larf: apply Householder reflector
//------------------------------------------------------------------------------

void SUITESPARSE_LAPACK_DLARF
(
    // input:
    const char *side,
    const SUITESPARSE_BLAS_INT *m,
    const SUITESPARSE_BLAS_INT *n,
    const double *V,
    const SUITESPARSE_BLAS_INT *incv,
    const double *tau,
    // input/output:
    double *C,
    // input:
    const SUITESPARSE_BLAS_INT *ldc,
    // workspace:
    double *Work
) ;

void SUITESPARSE_LAPACK_SLARF
(
    // input:
    const char *side,
    const SUITESPARSE_BLAS_INT *m,
    const SUITESPARSE_BLAS_INT *n,
    const float *V,
    const SUITESPARSE_BLAS_INT *incv,
    const float *tau,
    // input/output:
    float *C,
    // input:
    const SUITESPARSE_BLAS_INT *ldc,
    // workspace:
    float *Work
) ;

void SUITESPARSE_LAPACK_ZLARF
(
    // input:
    const char *side,
    const SUITESPARSE_BLAS_INT *m,
    const SUITESPARSE_BLAS_INT *n,
    const void *V,
    const SUITESPARSE_BLAS_INT *incv,
    const void *tau,
    // input/output:
    void *C,
    // input:
    const SUITESPARSE_BLAS_INT *ldc,
    // workspace:
    void *Work
) ;

void SUITESPARSE_LAPACK_CLARF
(
    // input:
    const char *side,
    const SUITESPARSE_BLAS_INT *m,
    const SUITESPARSE_BLAS_INT *n,
    const void *V,
    const SUITESPARSE_BLAS_INT *incv,
    const void *tau,
    // input/output:
    void *C,
    // input:
    const SUITESPARSE_BLAS_INT *ldc,
    // workspace:
    void *Work
) ;

#endif

//------------------------------------------------------------------------------
// macros for BLAS and SUITESPARSE_LAPACK functions
//------------------------------------------------------------------------------

#if defined ( SUITESPARSE_BLAS_MACROS )

#define SUITESPARSE_BLAS_dgemv(trans,m,n,alpha,A,lda,X,incx,beta,Y,incy,ok)   \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (M_blas_int, m, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (LDA_blas_int, lda, ok) ;                         \
    SUITESPARSE_TO_BLAS_INT (INCX_blas_int, incx, ok) ;                       \
    SUITESPARSE_TO_BLAS_INT (INCY_blas_int, incy, ok) ;                       \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_BLAS_DGEMV (trans, &M_blas_int, &N_blas_int, alpha, A,    \
            &LDA_blas_int, X, &INCX_blas_int, beta, Y, &INCY_blas_int) ;      \
    }                                                                         \
}

#define SUITESPARSE_BLAS_sgemv(trans,m,n,alpha,A,lda,X,incx,beta,Y,incy,ok)   \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (M_blas_int, m, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (LDA_blas_int, lda, ok) ;                         \
    SUITESPARSE_TO_BLAS_INT (INCX_blas_int, incx, ok) ;                       \
    SUITESPARSE_TO_BLAS_INT (INCY_blas_int, incy, ok) ;                       \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_BLAS_SGEMV (trans, &M_blas_int, &N_blas_int, alpha, A,    \
            &LDA_blas_int, X, &INCX_blas_int, beta, Y, &INCY_blas_int) ;      \
    }                                                                         \
}

#define SUITESPARSE_BLAS_zgemv(trans,m,n,alpha,A,lda,X,incx,beta,Y,incy,ok)   \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (M_blas_int, m, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (LDA_blas_int, lda, ok) ;                         \
    SUITESPARSE_TO_BLAS_INT (INCX_blas_int, incx, ok) ;                       \
    SUITESPARSE_TO_BLAS_INT (INCY_blas_int, incy, ok) ;                       \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_BLAS_ZGEMV (trans, &M_blas_int, &N_blas_int, alpha, A,    \
            &LDA_blas_int, X, &INCX_blas_int, beta, Y, &INCY_blas_int) ;      \
    }                                                                         \
}

#define SUITESPARSE_BLAS_cgemv(trans,m,n,alpha,A,lda,X,incx,beta,Y,incy,ok)   \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (M_blas_int, m, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (LDA_blas_int, lda, ok) ;                         \
    SUITESPARSE_TO_BLAS_INT (INCX_blas_int, incx, ok) ;                       \
    SUITESPARSE_TO_BLAS_INT (INCY_blas_int, incy, ok) ;                       \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_BLAS_CGEMV (trans, &M_blas_int, &N_blas_int, alpha, A,    \
            &LDA_blas_int, X, &INCX_blas_int, beta, Y, &INCY_blas_int) ;      \
    }                                                                         \
}

#define SUITESPARSE_BLAS_dtrsv(uplo,trans,diag,n,A,lda,X,incx,ok)             \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (LDA_blas_int, lda, ok) ;                         \
    SUITESPARSE_TO_BLAS_INT (INCX_blas_int, incx, ok) ;                       \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_BLAS_DTRSV (uplo, trans, diag, &N_blas_int, A,            \
            &LDA_blas_int, X, &INCX_blas_int) ;                               \
    }                                                                         \
}

#define SUITESPARSE_BLAS_strsv(uplo,trans,diag,n,A,lda,X,incx,ok)             \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (LDA_blas_int, lda, ok) ;                         \
    SUITESPARSE_TO_BLAS_INT (INCX_blas_int, incx, ok) ;                       \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_BLAS_STRSV (uplo, trans, diag, &N_blas_int, A,            \
            &LDA_blas_int, X, &INCX_blas_int) ;                               \
    }                                                                         \
}

#define SUITESPARSE_BLAS_ztrsv(uplo,trans,diag,n,A,lda,X,incx,ok)             \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (LDA_blas_int, lda, ok) ;                         \
    SUITESPARSE_TO_BLAS_INT (INCX_blas_int, incx, ok) ;                       \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_BLAS_ZTRSV (uplo, trans, diag, &N_blas_int, A,            \
            &LDA_blas_int, X, &INCX_blas_int) ;                               \
    }                                                                         \
}

#define SUITESPARSE_BLAS_ctrsv(uplo,trans,diag,n,A,lda,X,incx,ok)             \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (LDA_blas_int, lda, ok) ;                         \
    SUITESPARSE_TO_BLAS_INT (INCX_blas_int, incx, ok) ;                       \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_BLAS_CTRSV (uplo, trans, diag, &N_blas_int, A,            \
            &LDA_blas_int, X, &INCX_blas_int) ;                               \
    }                                                                         \
}

#define SUITESPARSE_BLAS_dtrsm(side,uplo,transa,diag,m,n,alpha,A,lda,B,ldb,ok)\
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (M_blas_int, m, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (LDA_blas_int, lda, ok) ;                         \
    SUITESPARSE_TO_BLAS_INT (LDB_blas_int, ldb, ok) ;                         \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_BLAS_DTRSM (side, uplo, transa, diag, &M_blas_int,        \
            &N_blas_int, alpha, A, &LDA_blas_int, B, &LDB_blas_int) ;         \
    }                                                                         \
}

#define SUITESPARSE_BLAS_strsm(side,uplo,transa,diag,m,n,alpha,A,lda,B,ldb,ok)\
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (M_blas_int, m, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (LDA_blas_int, lda, ok) ;                         \
    SUITESPARSE_TO_BLAS_INT (LDB_blas_int, ldb, ok) ;                         \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_BLAS_STRSM (side, uplo, transa, diag, &M_blas_int,        \
            &N_blas_int, alpha, A, &LDA_blas_int, B, &LDB_blas_int) ;         \
    }                                                                         \
}

#define SUITESPARSE_BLAS_ztrsm(side,uplo,transa,diag,m,n,alpha,A,lda,B,ldb,ok)\
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (M_blas_int, m, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (LDA_blas_int, lda, ok) ;                         \
    SUITESPARSE_TO_BLAS_INT (LDB_blas_int, ldb, ok) ;                         \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_BLAS_ZTRSM (side, uplo, transa, diag, &M_blas_int,        \
            &N_blas_int, alpha, A, &LDA_blas_int, B, &LDB_blas_int) ;         \
    }                                                                         \
}

#define SUITESPARSE_BLAS_ctrsm(side,uplo,transa,diag,m,n,alpha,A,lda,B,ldb,ok)\
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (M_blas_int, m, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (LDA_blas_int, lda, ok) ;                         \
    SUITESPARSE_TO_BLAS_INT (LDB_blas_int, ldb, ok) ;                         \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_BLAS_CTRSM (side, uplo, transa, diag, &M_blas_int,        \
            &N_blas_int, alpha, A, &LDA_blas_int, B, &LDB_blas_int) ;         \
    }                                                                         \
}

#define SUITESPARSE_BLAS_dgemm(transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,    \
    C,ldc,ok)                                                                 \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (M_blas_int, m, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (K_blas_int, k, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (LDA_blas_int, lda, ok) ;                         \
    SUITESPARSE_TO_BLAS_INT (LDB_blas_int, ldb, ok) ;                         \
    SUITESPARSE_TO_BLAS_INT (LDC_blas_int, ldc, ok) ;                         \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_BLAS_DGEMM (transa, transb, &M_blas_int, &N_blas_int,     \
            &K_blas_int, alpha, A, &LDA_blas_int, B, &LDB_blas_int, beta, C,  \
            &LDC_blas_int) ;                                                  \
    }                                                                         \
}

#define SUITESPARSE_BLAS_sgemm(transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,    \
    C,ldc,ok)                                                                 \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (M_blas_int, m, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (K_blas_int, k, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (LDA_blas_int, lda, ok) ;                         \
    SUITESPARSE_TO_BLAS_INT (LDB_blas_int, ldb, ok) ;                         \
    SUITESPARSE_TO_BLAS_INT (LDC_blas_int, ldc, ok) ;                         \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_BLAS_SGEMM (transa, transb, &M_blas_int, &N_blas_int,     \
            &K_blas_int, alpha, A, &LDA_blas_int, B, &LDB_blas_int, beta, C,  \
            &LDC_blas_int) ;                                                  \
    }                                                                         \
}

#define SUITESPARSE_BLAS_zgemm(transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,    \
    C,ldc,ok)                                                                 \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (M_blas_int, m, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (K_blas_int, k, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (LDA_blas_int, lda, ok) ;                         \
    SUITESPARSE_TO_BLAS_INT (LDB_blas_int, ldb, ok) ;                         \
    SUITESPARSE_TO_BLAS_INT (LDC_blas_int, ldc, ok) ;                         \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_BLAS_ZGEMM (transa, transb, &M_blas_int, &N_blas_int,     \
            &K_blas_int, alpha, A, &LDA_blas_int, B, &LDB_blas_int, beta, C,  \
            &LDC_blas_int) ;                                                  \
    }                                                                         \
}

#define SUITESPARSE_BLAS_cgemm(transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,    \
    C,ldc,ok)                                                                 \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (M_blas_int, m, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (K_blas_int, k, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (LDA_blas_int, lda, ok) ;                         \
    SUITESPARSE_TO_BLAS_INT (LDB_blas_int, ldb, ok) ;                         \
    SUITESPARSE_TO_BLAS_INT (LDC_blas_int, ldc, ok) ;                         \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_BLAS_CGEMM (transa, transb, &M_blas_int, &N_blas_int,     \
            &K_blas_int, alpha, A, &LDA_blas_int, B, &LDB_blas_int, beta, C,  \
            &LDC_blas_int) ;                                                  \
    }                                                                         \
}

#define SUITESPARSE_BLAS_dsyrk(uplo,trans,n,k,alpha,A,lda,beta,C,ldc,ok)      \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (K_blas_int, k, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (LDA_blas_int, lda, ok) ;                         \
    SUITESPARSE_TO_BLAS_INT (LDC_blas_int, ldc, ok) ;                         \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_BLAS_DSYRK (uplo, trans, &N_blas_int, &K_blas_int, alpha, \
            A, &LDA_blas_int, beta, C, &LDC_blas_int) ;                       \
    }                                                                         \
}

#define SUITESPARSE_BLAS_ssyrk(uplo,trans,n,k,alpha,A,lda,beta,C,ldc,ok)      \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (K_blas_int, k, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (LDA_blas_int, lda, ok) ;                         \
    SUITESPARSE_TO_BLAS_INT (LDC_blas_int, ldc, ok) ;                         \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_BLAS_SSYRK (uplo, trans, &N_blas_int, &K_blas_int, alpha, \
            A, &LDA_blas_int, beta, C, &LDC_blas_int) ;                       \
    }                                                                         \
}

#define SUITESPARSE_BLAS_zherk(uplo,trans,n,k,alpha,A,lda,beta,C,ldc,ok)      \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (K_blas_int, k, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (LDA_blas_int, lda, ok) ;                         \
    SUITESPARSE_TO_BLAS_INT (LDC_blas_int, ldc, ok) ;                         \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_BLAS_ZHERK (uplo, trans, &N_blas_int, &K_blas_int, alpha, \
            A, &LDA_blas_int, beta, C, &LDC_blas_int) ;                       \
    }                                                                         \
}

#define SUITESPARSE_BLAS_cherk(uplo,trans,n,k,alpha,A,lda,beta,C,ldc,ok)      \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (K_blas_int, k, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (LDA_blas_int, lda, ok) ;                         \
    SUITESPARSE_TO_BLAS_INT (LDC_blas_int, ldc, ok) ;                         \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_BLAS_CHERK (uplo, trans, &N_blas_int, &K_blas_int, alpha, \
            A, &LDA_blas_int, beta, C, &LDC_blas_int) ;                       \
    }                                                                         \
}

#define SUITESPARSE_LAPACK_dpotrf(uplo,n,A,lda,info,ok)                       \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (LDA_blas_int, lda, ok) ;                         \
    info = 1 ;                                                                \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_BLAS_INT LAPACK_Info = -999 ;                             \
        SUITESPARSE_LAPACK_DPOTRF (uplo, &N_blas_int, A, &LDA_blas_int,       \
          &LAPACK_Info) ;                                                     \
        info = (Int) LAPACK_Info ;                                            \
    }                                                                         \
}

#define SUITESPARSE_LAPACK_spotrf(uplo,n,A,lda,info,ok)                       \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (LDA_blas_int, lda, ok) ;                         \
    info = 1 ;                                                                \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_BLAS_INT LAPACK_Info = -999 ;                             \
        SUITESPARSE_LAPACK_SPOTRF (uplo, &N_blas_int, A, &LDA_blas_int,       \
          &LAPACK_Info) ;                                                     \
        info = (Int) LAPACK_Info ;                                            \
    }                                                                         \
}

#define SUITESPARSE_LAPACK_zpotrf(uplo,n,A,lda,info,ok)                       \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (LDA_blas_int, lda, ok) ;                         \
    info = 1 ;                                                                \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_BLAS_INT LAPACK_Info = -999 ;                             \
        SUITESPARSE_LAPACK_ZPOTRF (uplo, &N_blas_int, A, &LDA_blas_int,       \
            &LAPACK_Info) ;                                                   \
        info = LAPACK_Info ;                                                  \
    }                                                                         \
}

#define SUITESPARSE_LAPACK_cpotrf(uplo,n,A,lda,info,ok)                       \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (LDA_blas_int, lda, ok) ;                         \
    info = 1 ;                                                                \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_BLAS_INT LAPACK_Info = -999 ;                             \
        SUITESPARSE_LAPACK_CPOTRF (uplo, &N_blas_int, A, &LDA_blas_int,       \
            &LAPACK_Info) ;                                                   \
        info = LAPACK_Info ;                                                  \
    }                                                                         \
}

#define SUITESPARSE_BLAS_dscal(n,alpha,Y,incy,ok)                             \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (INCY_blas_int, incy, ok) ;                       \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_BLAS_DSCAL (&N_blas_int, alpha, Y, &INCY_blas_int) ;      \
    }                                                                         \
}

#define SUITESPARSE_BLAS_sscal(n,alpha,Y,incy,ok)                             \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (INCY_blas_int, incy, ok) ;                       \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_BLAS_SSCAL (&N_blas_int, alpha, Y, &INCY_blas_int) ;      \
    }                                                                         \
}

#define SUITESPARSE_BLAS_zscal(n,alpha,Y,incy,ok)                             \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (INCY_blas_int, incy, ok) ;                       \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_BLAS_ZSCAL (&N_blas_int, alpha, Y, &INCY_blas_int) ;      \
    }                                                                         \
}

#define SUITESPARSE_BLAS_cscal(n,alpha,Y,incy,ok)                             \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (INCY_blas_int, incy, ok) ;                       \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_BLAS_CSCAL (&N_blas_int, alpha, Y, &INCY_blas_int) ;      \
    }                                                                         \
}

#define SUITESPARSE_BLAS_dger(m,n,alpha,X,incx,Y,incy,A,lda,ok)               \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (M_blas_int, m, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (INCX_blas_int, incx, ok) ;                       \
    SUITESPARSE_TO_BLAS_INT (INCY_blas_int, incy, ok) ;                       \
    SUITESPARSE_TO_BLAS_INT (LDA_blas_int, lda, ok) ;                         \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_BLAS_DGER (&M_blas_int, &N_blas_int, alpha, X,            \
            &INCX_blas_int, Y, &INCY_blas_int, A, &LDA_blas_int) ;            \
    }                                                                         \
}

#define SUITESPARSE_BLAS_sger(m,n,alpha,X,incx,Y,incy,A,lda,ok)               \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (M_blas_int, m, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (INCX_blas_int, incx, ok) ;                       \
    SUITESPARSE_TO_BLAS_INT (INCY_blas_int, incy, ok) ;                       \
    SUITESPARSE_TO_BLAS_INT (LDA_blas_int, lda, ok) ;                         \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_BLAS_SGER (&M_blas_int, &N_blas_int, alpha, X,            \
            &INCX_blas_int, Y, &INCY_blas_int, A, &LDA_blas_int) ;            \
    }                                                                         \
}

#define SUITESPARSE_BLAS_zgeru(m,n,alpha,X,incx,Y,incy,A,lda,ok)              \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (M_blas_int, m, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (INCX_blas_int, incx, ok) ;                       \
    SUITESPARSE_TO_BLAS_INT (INCY_blas_int, incy, ok) ;                       \
    SUITESPARSE_TO_BLAS_INT (LDA_blas_int, lda, ok) ;                         \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_BLAS_ZGERU (&M_blas_int, &N_blas_int, alpha, X,           \
            &INCX_blas_int, Y, &INCY_blas_int, A, &LDA_blas_int) ;            \
    }                                                                         \
}

#define SUITESPARSE_BLAS_cgeru(m,n,alpha,X,incx,Y,incy,A,lda,ok)              \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (M_blas_int, m, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (INCX_blas_int, incx, ok) ;                       \
    SUITESPARSE_TO_BLAS_INT (INCY_blas_int, incy, ok) ;                       \
    SUITESPARSE_TO_BLAS_INT (LDA_blas_int, lda, ok) ;                         \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_BLAS_CGERU (&M_blas_int, &N_blas_int, alpha, X,           \
            &INCX_blas_int, Y, &INCY_blas_int, A, &LDA_blas_int) ;            \
    }                                                                         \
}

#define SUITESPARSE_LAPACK_dlarft(direct,storev,n,k,V,ldv,Tau,T,ldt,ok)       \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (K_blas_int, k, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (LDV_blas_int, ldv, ok) ;                         \
    SUITESPARSE_TO_BLAS_INT (LDT_blas_int, ldt, ok) ;                         \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_LAPACK_DLARFT (direct, storev, &N_blas_int, &K_blas_int,  \
            V, &LDV_blas_int, Tau, T, &LDT_blas_int) ;                        \
    }                                                                         \
}

#define SUITESPARSE_LAPACK_slarft(direct,storev,n,k,V,ldv,Tau,T,ldt,ok)       \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (K_blas_int, k, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (LDV_blas_int, ldv, ok) ;                         \
    SUITESPARSE_TO_BLAS_INT (LDT_blas_int, ldt, ok) ;                         \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_LAPACK_SLARFT (direct, storev, &N_blas_int, &K_blas_int,  \
            V, &LDV_blas_int, Tau, T, &LDT_blas_int) ;                        \
    }                                                                         \
}

#define SUITESPARSE_LAPACK_zlarft(direct,storev,n,k,V,ldv,Tau,T,ldt,ok)       \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (K_blas_int, k, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (LDV_blas_int, ldv, ok) ;                         \
    SUITESPARSE_TO_BLAS_INT (LDT_blas_int, ldt, ok) ;                         \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_LAPACK_ZLARFT (direct, storev, &N_blas_int, &K_blas_int,  \
            V, &LDV_blas_int, Tau, T, &LDT_blas_int) ;                        \
    }                                                                         \
}

#define SUITESPARSE_LAPACK_clarft(direct,storev,n,k,V,ldv,Tau,T,ldt,ok)       \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (K_blas_int, k, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (LDV_blas_int, ldv, ok) ;                         \
    SUITESPARSE_TO_BLAS_INT (LDT_blas_int, ldt, ok) ;                         \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_LAPACK_CLARFT (direct, storev, &N_blas_int, &K_blas_int,  \
            V, &LDV_blas_int, Tau, T, &LDT_blas_int) ;                        \
    }                                                                         \
}

#define SUITESPARSE_LAPACK_dlarfb(side,trans,direct,storev,m,n,k,V,ldv,T,ldt, \
    C,ldc,Work,ldwork,ok)                                                     \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (M_blas_int, m, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (K_blas_int, k, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (LDV_blas_int, ldv, ok) ;                         \
    SUITESPARSE_TO_BLAS_INT (LDT_blas_int, ldt, ok) ;                         \
    SUITESPARSE_TO_BLAS_INT (LDC_blas_int, ldc, ok) ;                         \
    SUITESPARSE_TO_BLAS_INT (LDWORK_blas_int, ldwork, ok) ;                   \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_LAPACK_DLARFB (side, trans, direct, storev, &M_blas_int,  \
            &N_blas_int, &K_blas_int, V, &LDV_blas_int, T, &LDT_blas_int, C,  \
            &LDC_blas_int, Work, &LDWORK_blas_int) ;                          \
    }                                                                         \
}

#define SUITESPARSE_LAPACK_slarfb(side,trans,direct,storev,m,n,k,V,ldv,T,ldt, \
    C,ldc,Work,ldwork,ok)                                                     \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (M_blas_int, m, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (K_blas_int, k, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (LDV_blas_int, ldv, ok) ;                         \
    SUITESPARSE_TO_BLAS_INT (LDT_blas_int, ldt, ok) ;                         \
    SUITESPARSE_TO_BLAS_INT (LDC_blas_int, ldc, ok) ;                         \
    SUITESPARSE_TO_BLAS_INT (LDWORK_blas_int, ldwork, ok) ;                   \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_LAPACK_SLARFB (side, trans, direct, storev, &M_blas_int,  \
            &N_blas_int, &K_blas_int, V, &LDV_blas_int, T, &LDT_blas_int, C,  \
            &LDC_blas_int, Work, &LDWORK_blas_int) ;                          \
    }                                                                         \
}

#define SUITESPARSE_LAPACK_zlarfb(side,trans,direct,storev,m,n,k,V,ldv,T,ldt, \
    C,ldc,Work,ldwork,ok)                                                     \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (M_blas_int, m, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (K_blas_int, k, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (LDV_blas_int, ldv, ok) ;                         \
    SUITESPARSE_TO_BLAS_INT (LDT_blas_int, ldt, ok) ;                         \
    SUITESPARSE_TO_BLAS_INT (LDC_blas_int, ldc, ok) ;                         \
    SUITESPARSE_TO_BLAS_INT (LDWORK_blas_int, ldwork, ok) ;                   \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_LAPACK_ZLARFB (side, trans, direct, storev, &M_blas_int,  \
            &N_blas_int, &K_blas_int, V, &LDV_blas_int, T, &LDT_blas_int, C,  \
            &LDC_blas_int, Work, &LDWORK_blas_int) ;                          \
    }                                                                         \
}

#define SUITESPARSE_LAPACK_clarfb(side,trans,direct,storev,m,n,k,V,ldv,T,ldt, \
    C,ldc,Work,ldwork,ok)                                                     \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (M_blas_int, m, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (K_blas_int, k, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (LDV_blas_int, ldv, ok) ;                         \
    SUITESPARSE_TO_BLAS_INT (LDT_blas_int, ldt, ok) ;                         \
    SUITESPARSE_TO_BLAS_INT (LDC_blas_int, ldc, ok) ;                         \
    SUITESPARSE_TO_BLAS_INT (LDWORK_blas_int, ldwork, ok) ;                   \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_LAPACK_CLARFB (side, trans, direct, storev, &M_blas_int,  \
            &N_blas_int, &K_blas_int, V, &LDV_blas_int, T, &LDT_blas_int, C,  \
            &LDC_blas_int, Work, &LDWORK_blas_int) ;                          \
    }                                                                         \
}

#define SUITESPARSE_BLAS_dnrm2(result,n,X,incx,ok)                            \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (INCX_blas_int, incx, ok) ;                       \
    result = 0 ;                                                              \
    if (ok)                                                                   \
    {                                                                         \
        result = SUITESPARSE_BLAS_DNRM2 (&N_blas_int, X, &INCX_blas_int) ;    \
    }                                                                         \
}

#define SUITESPARSE_BLAS_snrm2(result,n,X,incx,ok)                            \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (INCX_blas_int, incx, ok) ;                       \
    result = 0 ;                                                              \
    if (ok)                                                                   \
    {                                                                         \
        result = SUITESPARSE_BLAS_SNRM2 (&N_blas_int, X, &INCX_blas_int) ;    \
    }                                                                         \
}

#define SUITESPARSE_BLAS_dznrm2(result,n,X,incx,ok)                           \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (INCX_blas_int, incx, ok) ;                       \
    result = 0 ;                                                              \
    if (ok)                                                                   \
    {                                                                         \
        result = SUITESPARSE_BLAS_DZNRM2 (&N_blas_int, X, &INCX_blas_int) ;   \
    }                                                                         \
}

#define SUITESPARSE_BLAS_scnrm2(result,n,X,incx,ok)                           \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (INCX_blas_int, incx, ok) ;                       \
    result = 0 ;                                                              \
    if (ok)                                                                   \
    {                                                                         \
        result = SUITESPARSE_BLAS_SCNRM2 (&N_blas_int, X, &INCX_blas_int) ;   \
    }                                                                         \
}

#define SUITESPARSE_LAPACK_dlarfg(n,alpha,X,incx,tau,ok)                      \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (INCX_blas_int, incx, ok) ;                       \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_LAPACK_DLARFG (&N_blas_int, alpha, X, &INCX_blas_int,     \
            tau) ;                                                            \
    }                                                                         \
}

#define SUITESPARSE_LAPACK_slarfg(n,alpha,X,incx,tau,ok)                      \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (INCX_blas_int, incx, ok) ;                       \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_LAPACK_SLARFG (&N_blas_int, alpha, X, &INCX_blas_int,     \
            tau) ;                                                            \
    }                                                                         \
}

#define SUITESPARSE_LAPACK_zlarfg(n,alpha,X,incx,tau,ok)                      \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (INCX_blas_int, incx, ok) ;                       \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_LAPACK_ZLARFG (&N_blas_int, alpha, X, &INCX_blas_int,     \
            tau) ;                                                            \
    }                                                                         \
}

#define SUITESPARSE_LAPACK_clarfg(n,alpha,X,incx,tau,ok)                      \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (INCX_blas_int, incx, ok) ;                       \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_LAPACK_CLARFG (&N_blas_int, alpha, X, &INCX_blas_int,     \
            tau) ;                                                            \
    }                                                                         \
}

#define SUITESPARSE_LAPACK_dlarf(side,m,n,V,incv,tau,C,ldc,Work,ok)           \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (M_blas_int, m, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (INCV_blas_int, incv, ok) ;                       \
    SUITESPARSE_TO_BLAS_INT (LDC_blas_int, ldc, ok) ;                         \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_LAPACK_DLARF (side, &M_blas_int, &N_blas_int, V,          \
            &INCV_blas_int, tau, C, &LDC_blas_int, Work) ;                    \
    }                                                                         \
}

#define SUITESPARSE_LAPACK_slarf(side,m,n,V,incv,tau,C,ldc,Work,ok)           \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (M_blas_int, m, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (INCV_blas_int, incv, ok) ;                       \
    SUITESPARSE_TO_BLAS_INT (LDC_blas_int, ldc, ok) ;                         \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_LAPACK_SLARF (side, &M_blas_int, &N_blas_int, V,          \
            &INCV_blas_int, tau, C, &LDC_blas_int, Work) ;                    \
    }                                                                         \
}

#define SUITESPARSE_LAPACK_zlarf(side,m,n,V,incv,tau,C,ldc,Work,ok)           \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (M_blas_int, m, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (INCV_blas_int, incv, ok) ;                       \
    SUITESPARSE_TO_BLAS_INT (LDC_blas_int, ldc, ok) ;                         \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_LAPACK_ZLARF (side, &M_blas_int, &N_blas_int, V,          \
            &INCV_blas_int, tau, C, &LDC_blas_int, Work) ;                    \
    }                                                                         \
}

#define SUITESPARSE_LAPACK_clarf(side,m,n,V,incv,tau,C,ldc,Work,ok)           \
{                                                                             \
    SUITESPARSE_TO_BLAS_INT (M_blas_int, m, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (N_blas_int, n, ok) ;                             \
    SUITESPARSE_TO_BLAS_INT (INCV_blas_int, incv, ok) ;                       \
    SUITESPARSE_TO_BLAS_INT (LDC_blas_int, ldc, ok) ;                         \
    if (ok)                                                                   \
    {                                                                         \
        SUITESPARSE_LAPACK_CLARF (side, &M_blas_int, &N_blas_int, V,          \
            &INCV_blas_int, tau, C, &LDC_blas_int, Work) ;                    \
    }                                                                         \
}
#endif

//------------------------------------------------------------------------------
// SuiteSparse_BLAS_library: return name of BLAS library found
//------------------------------------------------------------------------------

// Returns the name of the BLAS library found by SuiteSparse_config

const char *SuiteSparse_BLAS_library ( void ) ;

//------------------------------------------------------------------------------
// SuiteSparse_BLAS_integer_size: return sizeof (SUITESPARSE_BLAS_INT)
//------------------------------------------------------------------------------

size_t SuiteSparse_BLAS_integer_size ( void ) ;

#ifdef __cplusplus
}
#endif
#endif

