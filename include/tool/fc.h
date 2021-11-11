#pragma once
#include "macro.h"
#include <string>
#include <tinker/routines.h>

// version
std::string tinker_f_version(std::string infile, std::string status);

// file
void tinker_f_rewind(int* unit);

void tinker_f_close(int* unit);

void tinker_f_open(int* unit, std::string file, std::string status);

// memory
int tinker_f_allocated(void* p);

void tinker_f_deallocate(void* p);

void tinker_f_allocate_byte(void** pp, size_t bytes);

template <class T>
void tinker_f_allocate_dim(T** pp, int dim)
{
   void** p = (void**)pp;
   size_t bytes = sizeof(T) * dim;
   tinker_f_allocate_byte(p, bytes);
}

// read stdin
std::string tinker_f_read_stdin_line();
