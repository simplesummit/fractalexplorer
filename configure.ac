###  configure.ac -- Autoconf configuration script
#
#    This file is part of the fractalexplorer project
#
#    fractalexplorer source code, as well as any other resources in this
#  project are free software; you are free to redistribute it and/or modify
#  them under the terms of the GNU General Public License; either version 3 of
#  thelicense, or any later version.
#
#    These programs are hopefully useful and reliable, but it is understood
#  that these are provided WITHOUT ANY WARRANTY, or MERCHANTABILITY or FITNESS
#  FOR A PARTICULAR PURPOSE. See the GPLv3 or email at
#  <cade@cade.site> for more info on this.
#
#    Here is a copy of the GPL v3, which this software is licensed under. You
#  can also find a copy at http://www.gnu.org/licenses/.
###


AC_PREREQ([2.55])
AC_INIT([fractalexplorer], [0.1.1], [cade@cade.site])

AC_CONFIG_MACRO_DIRS([custom_m4])

AM_INIT_AUTOMAKE([])


AC_CONFIG_SRCDIR([src/fractalexplorer.h])
AC_CONFIG_HEADERS([config.h])

if test -z "$CC"; then
    CC=mpicc
fi

AC_LANG(C)
AC_PROG_CC
PKG_PROG_PKG_CONFIG

AC_CHECK_HEADERS([stdlib.h stdio.h stdbool.h string.h math.h sys/time.h sys/stat.h time.h unistd.h assert.h stdarg.h getopt.h ctype.h dlfcn.h])

AC_CHECK_LIB(m, log)


AC_ARG_WITH([lz4],
    [AS_HELP_STRING([--with-lz4],
    [use lz4 @<:@default=yes@:>@])],
    [],
    [with_lz4=yes]
)

if test "x$with_lz4" != xno; then
    AS_IF([test "x$with_lz4" != xyes && test "x$with_lz4" != xcheck], [
        LDFLAGS="$LDFLAGS -L$with_lz4/lib"
        CFLAGS="$CFLAGS -I$with_lz4/include"
    ])

    AC_CHECK_LIB([lz4], [LZ4_compress_default], [
        AC_CHECK_HEADERS([lz4.h])
        AC_DEFINE([HAVE_LZ4], [1], [Define if you have lz4])
        export with_lz4=yes
        LIBS="$LIBS -llz4"
        AC_MSG_CHECKING([liblz4])
        AC_MSG_RESULT([yes])

    ], [
        AC_MSG_CHECKING([liblz4])
        AC_MSG_RESULT([no])
        if test "x$with_lz4" != xcheck; then
            AC_MSG_FAILURE([--with-lz4 was given, but test for liblz4 failed])
        fi
        export with_lz4=no
    ])
fi


AC_ARG_WITH([cuda],
    [AS_HELP_STRING([--with-cuda],
    [use cuda @<:@default=check@:>@])],
    [],
    [with_cuda=check]
)

cuda_prefix="$with_cuda"

# Setting the prefix to the default if only --with-cuda was given
if test "$cuda_prefix" == "yes"; then
	cuda_prefix="/usr/local/cuda"
fi

if test "$cuda_prefix" == "check"; then
	cuda_prefix="/usr/local/cuda"
fi


if test "x${with_cuda}" != xno; then
  AS_IF([test "x${with_cuda}" != xyes && test "x${with_cuda}" != xcheck], [
    LDFLAGS="$LDFLAGS -L${with_cuda}/lib$SUFFIX -Wl,-rpath,${with_cuda}/lib$SUFFIX"
    CFLAGS="$CFLAGS -I${with_cuda}/include"
  ])
  AS_IF([test "x${with_cuda}" == xcheck], [
    LDFLAGS="$LDFLAGS -L/usr/local/cuda/lib$SUFFIX -Wl,-rpath,${with_cuda}/lib$SUFFIX"
    CFLAGS="$CFLAGS -I/usr/local/cuda/include"
  ])
fi


VALID_CUDA=no

# Checking for nvcc
AC_MSG_CHECKING([nvcc in $cuda_prefix/bin])
if test -x "$cuda_prefix/bin/nvcc"; then
	AC_MSG_RESULT([found])
	AC_DEFINE_UNQUOTED([NVCC_PATH], ["$cuda_prefix/bin/nvcc"], [Path to nvcc binary])
	# We need to add the CUDA search directories for header and lib searches

	CUDA_CFLAGS=""
  AC_DEFINE([HAVE_CUDA], [], [Have cuda support])

	# Saving the current flags
	ax_save_CFLAGS="${CFLAGS}"
	ax_save_LDFLAGS="${LDFLAGS}"

	# Announcing the new variables
	AC_SUBST([CUDA_CFLAGS])
	AC_SUBST([CUDA_LDFLAGS])
	AC_SUBST([NVCC],[$cuda_prefix/bin/nvcc])
  AC_CHECK_FILE([$cuda_prefix/lib64],[lib64_found=yes],[lib64_found=no])
  AC_CHECK_FILE([$cuda_prefix/lib32],[lib32_found=yes],[lib32_found=no])
  AC_CHECK_FILE([$cuda_prefix/lib],[lib_found=yes],[lib_found=no])
	if test "x$lib64_found" = xyes ; then
			AC_SUBST([CUDA_LIBDIR],[$cuda_prefix/lib64])
      CUDA_LIBDIR=$cuda_prefix/lib64
      VALID_CUDA=yes
	else
    if test "x$lib_found" = xyes ; then
  			AC_SUBST([CUDA_LIBDIR],[$cuda_prefix/lib])
        CUDA_LIBDIR=$cuda_prefix/lib
        VALID_CUDA=yes
  	else
      if test "x$lib32_found" = xyes ; then
    			AC_SUBST([CUDA_LIBDIR],[$cuda_prefix/lib32])
          CUDA_LIBDIR=$cuda_prefix/lib32
          VALID_CUDA=yes
    	else
        AC_MSG_WARN([Couldn't find cuda lib directory])
        VALID_CUDA=no
      fi
    fi
  fi

	if test "x$VALID_CUDA" != xno ; then
    echo $CUDA_LIBDIR
		CUDA_CFLAGS="$CUDA_CFLAGS -I$cuda_prefix/include/"
		CFLAGS="$CUDA_CFLAGS $CFLAGS"
		CUDA_LDFLAGS="-L$CUDA_LIBDIR"
		LDFLAGS="$CUDA_LDFLAGS $LDFLAGS -Wl,-rpath,$CUDA_LIBDIR -lcuda -lcudart -lcudadevrt"

		# And the header and the lib
		AC_CHECK_HEADERS([cuda.h cuda_runtime.h helper_functions.h helper_cuda.h])
		if test "x$VALID_CUDA" != "xno" ; then
			AC_CHECK_LIB([cuda], [cuInit], [VALID_CUDA=yes], AC_MSG_WARN([Couldn't find libcuda]
			VALID_CUDA=no))
		fi
  else
    # Returning to the original flags
    CFLAGS="${ax_save_CFLAGS} -I$cuda_prefix/include"
    LDFLAGS="${ax_save_LDFLAGS} -L$cuda_prefix/lib$SUFFIX -Wl,-rpath,$cuda_prefix/lib$SUFFIX"
	fi
else
	AC_MSG_RESULT([not found!])
	AC_MSG_WARN([nvcc was not found in $cuda_prefix/bin])
	VALID_CUDA=no
fi

if test "x$with_cuda" = xyes && test x$VALID_CUDA = xyes ; then
  AC_DEFINE([USE_CUDA], [], [use cuda])
	AC_MSG_NOTICE([Building with CUDA bindings])
elif test "x$with_cuda" = xyes && test x$VALID_CUDA = xno ; then
	AC_MSG_ERROR([Cannot build CUDA bindings. Check errors])
fi


AC_ARG_WITH([sdl2],
    [AS_HELP_STRING([--with-sdl2],
    [use sdl2 @<:@default=check@:>@])],
    [],
    [with_sdl2=check]
)


if test "x$with_sdl2" != xno; then
    AS_IF([test "x$with_sdl2" != xyes && test "x$with_sdl2" != xcheck], [
        LDFLAGS="$LDFLAGS -L$with_sdl2/lib"
        CFLAGS="$CFLAGS -I$with_sdl2/include"
    ])

    SDL_VERSION=2.0.0
    AM_PATH_SDL2([$SDL_VERSION],[
            AC_MSG_CHECKING([sdl2])
            AC_MSG_RESULT([yes])
            export with_sdl2=yes
        ], [
            AC_MSG_CHECKING([sdl2])
            AC_MSG_RESULT([no])
            if test "x$with_sdl2" != xcheck; then
                AC_MSG_FAILURE([--with-sdl2 was given, but test for libsdl failed])
            fi
            export with_sdl2=no
        ]
    )
fi

AC_SUBST([SDL_CFLAGS], [$SDL_CFLAGS])
AC_SUBST([SDL_LDFLAGS], [$SDL_LIBS])


AM_CONDITIONAL([HAVE_CUDA], [test x$VALID_CUDA == xyes])
AM_CONDITIONAL([HAVE_SDL], [test x$with_sdl2 != xno])


AC_CHECK_SIZEOF([double])
AC_CHECK_SIZEOF([unsigned char])
AC_CHECK_SIZEOF([unsigned short])
AC_CHECK_SIZEOF([unsigned int])
AC_CHECK_SIZEOF([unsigned long])

# Checks for library functions.
#AC_FUNC_MALLOC
#AC_FUNC_REALLOC

# todo, add tests
AC_CONFIG_FILES([Makefile src/Makefile])
AC_OUTPUT
