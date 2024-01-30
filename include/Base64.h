#pragma once

#ifndef BASE_64_H
#define BASE_64_H

#include <string>

std::string base64_encode(const unsigned char* bytes_to_encode, unsigned int in_len);
std::string base64_decode(std::string const& s);

#endif
