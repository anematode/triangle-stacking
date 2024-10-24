#include "image.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-field-initializers"

#include "3rdparty/stb_image.h"
#include "3rdparty/stb_image_write.h"

#pragma clang diagnostic pop
uint32_t as_uint(const float x) {
  uint32_t a;
  memcpy(&a, &x, sizeof(a));
  return a;
}

float as_float(const uint32_t x) {
  float a;
  memcpy(&a, &x, sizeof(a));
  return a;
}

float half_to_float(const unsigned short x) { // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
  const uint32_t e = (x&0x7C00)>>10; // exponent
  const uint32_t m = (x&0x03FF)<<13; // mantissa
  const uint32_t v = as_uint((float)m)>>23; // evil log2 bit hack to count leading zeros in denormalized format
  return as_float((x&0x8000)<<16 | (e!=0)*((e+112)<<23|m) | ((e==0)&(m!=0))*((v-37)<<23|((m<<(150-v))&0x007FE000))); // sign : normalized : denormalized
}

unsigned short float_to_half(const float x) { // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
  const uint32_t b = as_uint(x)+0x00001000; // round-to-nearest-even: add last bit after truncated mantissa
  const uint32_t e = (b&0x7F800000)>>23; // exponent
  const uint32_t m = b&0x007FFFFF; // mantissa; in line below: 0x007FF000 = 0x00800000-0x00001000 = decimal indicator flag - initial rounding
  return (b&0x80000000)>>16 | (e>112)*((((e-112)<<10)&0x7C00)|m>>13) | ((e<113)&(e>101))*((((0x007FF000+m)>>(125-e))+1)>>1) | (e>143)*0x7FFF; // sign : normalized : denormalized : saturate
}

#ifdef SFML_SUPPORTED
bool poll_events(sf::RenderWindow &window, bool forever) {
  do {
    if (sf::Event event; window.pollEvent(event)) {
      if (event.type == sf::Event::Closed) {
        window.close();
        return true;
      }
    }
  } while (forever && window.isOpen());
  return false;
}
#endif