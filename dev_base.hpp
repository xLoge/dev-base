#include <vector>
#include <cstring>
#include <string>
#include <utility>
#include <iterator>
#include <type_traits>
#include <immintrin.h>

// DATA TYPES
namespace db
{
	typedef signed char int8_t;
	typedef unsigned char uint8_t;

	typedef signed short int16_t;
	typedef unsigned short uint16_t;

	typedef signed int int32_t;
	typedef unsigned int uint32_t;

	typedef signed long long int64_t;
	typedef unsigned long long uint64_t;

	typedef unsigned long long size_t;

	template<bool Test, class If_Test, class Else>
	struct enable_if_else {
		using type = Else;
	};

	template <class If_Test, class Else>
	struct enable_if_else<true, If_Test, Else> {
		using type = If_Test;
	};
}

// MEM OPERATIONS
namespace db
{
	template <class Ty>
	constexpr inline Ty* raw_memcpy(Ty* _dst, const Ty* _src, db::size_t _size) noexcept
	{
		while (_size--) { *_dst++ = *_src++; }
		return _dst;
	}

	// * constexpr memcpy for any type of char
	template <class Ty>
	constexpr inline Ty* memcpy(Ty* _dst, const Ty* const _src, const db::size_t _size) noexcept
	{
		if (std::is_constant_evaluated()) {
			return db::raw_memcpy<Ty>(_dst, _src, _size);
		}
		else {
			return static_cast<Ty*>(std::memcpy(_dst, _src, _size * sizeof(Ty)));
		}
	}

	template <class Ty>
	constexpr inline db::int32_t raw_memcmp(const Ty* const _buff1, const Ty* const _buff2, db::size_t _size) noexcept
	{
		for (; 0 < _size; ++_buff1, ++_buff2, --_size) {
			if (*_buff1 != *_buff2) {
				return *_buff1 < *_buff2 ? -1 : 1;
			}
		}
		return 0;
	}

	template <class Ty>
	constexpr inline db::int32_t memcmp(const Ty* const _buff1, const Ty* const _buff2, db::size_t _size) noexcept
	{
		if (std::is_constant_evaluated()) {
			return db::raw_memcmp(_buff1, _buff2, _size);
		}
		else {
			return std::memcmp(_buff1, _buff2, _size);
		}
	}

	// * constexpr memchr for any type of char but reversed
	template <class Ty>
	constexpr inline const Ty* raw_rmemchr(const Ty* _str, const Ty _ch, const db::size_t _len)
	{
		const Ty* _end = _str;
		_str += _len;
		while (_str-- != _end) {
			if (*_str == _ch) {
				return _str;
			}
		}
		return nullptr;
	}

	// * constexpr memchr for any type of char but reversed
	template <class Ty>
	constexpr inline const Ty* rmemchr(const Ty* _str, const Ty _ch, const db::size_t _len)
	{
		return db::raw_rmemchr(_str, _ch, _len);
	}

	template <class Ty>
	constexpr inline const Ty* raw_memchr(const Ty* _str, const Ty _ch, const db::size_t _maxlen) noexcept
	{
		while (*_str) {
			if (*_str++ == _ch) {
				return --_str;
			}
		}
		return nullptr;
	}

	// * constexpr memchr for any type of char
	template <class Ty>
	constexpr inline const Ty* memchr(const Ty* _str, const Ty _ch, const db::size_t _maxlen) noexcept
	{
		if (std::is_constant_evaluated()) {
			return db::raw_memchr<Ty>(_str, _ch, _maxlen);
		}
		else {
			return static_cast<const Ty*>(std::memchr(_str, _ch, _maxlen));
		}
	}
}

// STRING OPERATIONS
namespace db
{
	// * fast strlen version I wrote with simd instructions
	template <class char_type>
	inline db::size_t fast_strlen_simd(const char_type* _begin)
	{
		const char_type* end = _begin;

		// skip 32 byte chunks
		for (;;) {
			if (_mm256_movemask_epi8(
					_mm256_cmpeq_epi8(*reinterpret_cast<const __m256i*>(end), _mm256_set1_epi8(0))
				))
			{
				break;
			}
			end += 32U / sizeof(char_type);
		}

		//skip last 16 byte chunk
		if (!_mm_movemask_epi8(
				_mm_cmpeq_epi8(*reinterpret_cast<const __m128i*>(end), _mm_set1_epi8(0))
			))
		{
			end += 16U / sizeof(char_type);
		}

		// skip last 8 byte chunk 
		constexpr db::size_t mask_high = static_cast<db::size_t>(0x8080808080808080U); // Works for X64 and X86
		constexpr db::size_t mask_low = static_cast<db::size_t>(0x0101010101010101U);
		const db::size_t data = *reinterpret_cast<const db::size_t*>(end);
		
		if (((data - mask_low) & (~data) & mask_high) == 0) {
			end += 8U / sizeof(char_type);
		}

		// count rest one by one
		for (; *end != char_type(); ++end);

		return static_cast<db::size_t>(end - _begin);
	}

	// * fast strlen version I wrote
	template <class char_type>
	constexpr inline db::size_t fast_strlen(const char_type* _begin)
	{
		constexpr db::size_t mask_high = static_cast<db::size_t>(0x8080808080808080U); // Works for X64 and X86
		constexpr db::size_t mask_low  = static_cast<db::size_t>(0x0101010101010101U);
		const db::size_t* aligned_end = reinterpret_cast<const db::size_t*>(_begin);
		const char_type* end = _begin;

		// Check 8 bytes at once without simd instructions
		for (db::size_t data;;) {
			data = *aligned_end++;
			if ((data - mask_low) & (~data) & mask_high) {
				break;
			}
			end = reinterpret_cast<const char_type*>(aligned_end);
		}

		// Count rest one by one
		for (; *end != char_type(); ++end);

		return static_cast<db::size_t>(end - _begin);
	}

	// * constexpr strlen base for any char type
	template <class char_type>
	constexpr inline db::size_t raw_strlen(const char_type* _begin) noexcept
	{
		const char_type* end = _begin;
		for (; *end; ++end);
		return (end - _begin);
	}

	// * constexpr strlen for any type of char
	template <class char_type>
	constexpr inline db::size_t strlen(const char_type* _begin)
	{
		if constexpr (std::is_same_v<char, char_type>
#ifdef	__cpp_char8_t
			|| std::is_same_v<char8_t, char_type>) {
#else
			) {
#endif 
			return std::strlen(reinterpret_cast<const char*>(_begin));
		}

		if constexpr (std::is_same_v<wchar_t, char_type> || std::is_same_v<char16_t, char_type>) {
			return std::wcslen(reinterpret_cast<const wchar_t*>(_begin));
		}

		if constexpr (std::is_same_v<char32_t, char_type>) {
			return db::raw_strlen<char32_t>(_begin);
		}

		throw std::invalid_argument("no type of char specified.");
	}

	// * guesses base of numeric string
	template <class char_type>
	constexpr inline db::uint16_t guess_base(const char_type* _str) noexcept
	{
		if (_str[1] == 'x' || _str[1] == 'X') {
			return 16;
		}

		bool _maybe_base_2 = true;

		if (_str[1] == 'b' || _str[1] == 'B') {
			_str += 2;
		}

		while (*_str) {
			if (*_str < '0') {
				return 0;
			}
			else if (*_str > '9')
			{
				if (*_str >= 'a' && *_str <= 'f' || *_str >= 'A' && *_str <= 'F') {
					return 16;
				}
				else {
					return 0;
				}
			}
			else if (*_str > '2') {
				_maybe_base_2 = false;
			}
			++_str;
		}

		if (_maybe_base_2) {
			return 2;
		}
		else {
			return 10;
		}
	}
}

// NUMBER OPERATIONS / MATH
namespace db
{
	// * Signed to Unsigned number
	template <class From, class To = std::make_unsigned_t<From>>
	constexpr inline To to_unsigned(const From _num) noexcept
	{
		constexpr bool same_type = std::is_same_v<From, To>;
		constexpr bool can_negative = !std::is_unsigned_v<From>;
#if _HAS_CXX17
		if constexpr (can_negative && !same_type) {
#else
		if (can_negative) {
#endif
			if (_num < 0) {
				return (static_cast<From>(0) - static_cast<To>(_num));
			}
		}
		return _num;
		}

	// * Unsigned to Signed number
	template <class To = db::int64_t, class From>
	constexpr inline To to_signed(const From _num) noexcept
	{
		constexpr bool is_unsigned = std::is_unsigned_v<From>;
#if _HAS_CXX17
		if constexpr(is_unsigned) {
#else 
		if (is_unsigned) {
#endif
			return static_cast<To>(_num);
		}

		return _num;
	}

	// * square root using simd instruction
	inline double sqrt_simd(const double _val) noexcept
	{
		const auto val = _mm_sqrt_pd(_mm_set_sd(_val));
		return _mm_cvtsd_f64(val);
	}

	// * inverse square root using simd instruction
	inline float inv_sqrt_simd(const float _val) noexcept
	{
		const auto val = _mm_rsqrt_ss(_mm_set_ss(_val));
		return _mm_cvtss_f32(val);
	}

	// * pow using simd instruction
	inline double pow_simd(const double _base, const double _expo) noexcept
	{
		const auto val = _mm_pow_pd(_mm_set_sd(_base), _mm_set_sd(_expo));
		return _mm_cvtsd_f64(val);
	}

	// * pow mod using simd instructions
	inline double powm_simd(const double _base, const double _expo, const double _mod) noexcept
	{
		const auto val = _mm_fmod_pd(
			_mm_pow_pd(_mm_set_sd(_base), _mm_set_sd(_expo)),
			_mm_set_sd(_mod));
		return _mm_cvtsd_f64(val);
	}

	// * atan using simd instruction
	inline double atan_simd(const double _X) noexcept
	{
		const auto val = _mm_atan_pd(_mm_set_sd(_X));
		return _mm_cvtsd_f64(val);
	}
	
	// * atan2 using simd instruction
	inline double atan2_simd(const double _X, const double _Y) noexcept
	{
		const auto val = _mm_atan2_pd(_mm_set_sd(_X), _mm_set_sd(_Y));
		return _mm_cvtsd_f64(val);
	}

	// * sin using simd instruction
	inline double sin_simd(const double _val) noexcept
	{
		const auto val = _mm_sin_pd(_mm_set_sd(_val));
		return _mm_cvtsd_f64(val);
	}

	// * cos using simd instruction
	inline double cos_simd(const double _val) noexcept
	{
		const auto val = _mm_cos_pd(_mm_set_sd(_val));
		return _mm_cvtsd_f64(val);
	}

	// * tan using simd instruction
	inline double tan_simd(const double _val) noexcept
	{
		const auto val = _mm_tan_pd(_mm_set_sd(_val));
		return _mm_cvtsd_f64(val);
	}

	// * tand using simd instruction
	inline double tand_simd(const double _val) noexcept
	{
		const auto val = _mm_tand_pd(_mm_set_sd(_val));
		return _mm_cvtsd_f64(val);
	}

	// * tanh using simd instruction
	inline double tanh_simd(const double _val) noexcept
	{
		const auto val = _mm_tanh_pd(_mm_set_sd(_val));
		return _mm_cvtsd_f64(val);
	}

	// * acos using simd instruction
	inline double acos_simd(const double _val) noexcept
	{
		const auto val = _mm_acos_pd(_mm_set_sd(_val));
		return _mm_cvtsd_f64(val);
	}

	// * asin using simd instruction
	inline double asin_simd(const double _val) noexcept
	{
		const auto val = _mm_asin_pd(_mm_set_sd(_val));
		return _mm_cvtsd_f64(val);
	}
	
	// * hypot using simd instruction
	inline double hypot_simd(const double _X, const double _Y) noexcept
	{
		const auto val = _mm_hypot_pd(_mm_set_sd(_X), _mm_set_sd(_Y));
		return _mm_cvtsd_f64(val);
	}

	// * log using simd instruction
	inline double log_simd(const double _val) noexcept
	{
		const auto val = _mm_log_pd(_mm_set_sd(_val));
		return _mm_cvtsd_f64(val);
	}

	// * log2 using simd instruction
	inline double log2_simd(const double _val) noexcept
	{
		const auto val = _mm_log2_pd(_mm_set_sd(_val));
		return _mm_cvtsd_f64(val);
	}

	// * log10 using simd instruction
	inline double log10_simd(const double _val) noexcept
	{
		const auto val = _mm_log10_pd(_mm_set_sd(_val));
		return _mm_cvtsd_f64(val);
	}

	template<class Base, class Expo>
	constexpr inline auto pow(Base _base, Expo _expo)
	{
		using result = db::enable_if_else <std::is_integral_v<Base>,
			typename db::enable_if_else<std::is_unsigned_v<Base>,
			db::uint64_t,
			db::int64_t>::type,
			Base
		>::type;

		result res = 1;

		while (_expo > 0) {
			if (_expo & 1) {
				res *= _base;
			}
			_base *= _base;
			_expo >>= 1;
		}

		return res;
	}

	template<class Base, class Expo, class Mod>
	constexpr inline auto powm(Base _base, Expo _expo, const Mod& _mod)
	{
		using result = db::enable_if_else <std::is_integral_v<Base>,
			typename db::enable_if_else<std::is_unsigned_v<Base>,
			db::uint64_t,
			db::int64_t>::type,
			Base
		>::type;

		result res = 1;

		while (_expo > 0) {
			if (_expo & 1) {
				res = (res * _base) % _mod;
			}
			_base = (_base * _base) % _mod;
			_expo >>= 1;
		}

		return res;
	}
}

// NUMBER TO STRING
namespace db
{
	// * Floatingpoint to string
	template <db::size_t _PRECISION = 5, class char_type = char, class Ty = float>
	constexpr inline std::basic_string<char_type> ftos(Ty _num)
	{
		char_type buff[21 + _PRECISION];
		char_type* begin = buff;

		// Negative check
		if (_num < 0.0) {
			*begin++ = '-';
			_num *= -1.0;
		}

		// Round
		static const auto prec1 = std::pow(0.1, _PRECISION);
		const double tmp_num = _num / prec1;
		_num = (tmp_num >= 0.5 ? (tmp_num + 1.0) : tmp_num) * prec1;

		// Get first number
		db::size_t first = static_cast<db::size_t>(_num);
		_num -= first;
		_num *= 10.0;

		char_type* beg = begin;

		// Set First
		while (first >= 1) {
			*begin++ = static_cast<char_type>(first % 10 + '0');
			first /= 10;
		}

		std::reverse(beg, begin);

		if (_num) {
			*begin++ = '.';

			db::size_t maxlen = _PRECISION;

			while (_num && maxlen--)
			{
				first = static_cast<db::size_t>(_num);
				_num -= first;
				_num *= 10.0;
				*begin++ = static_cast<char_type>(first + '0');
			}
		}

		return std::basic_string<char_type>(buff, static_cast<db::size_t>(begin - buff));
	}

	// * Number to binary string
	template <class char_type = char, class Ty = db::int32_t, bool FULL_OUTPUT = true>
	constexpr inline std::basic_string<char_type> ntobs(const Ty _num) noexcept
	{
		constexpr auto can_negative = !std::is_unsigned_v<Ty>;
		constexpr auto buff_size = std::numeric_limits<Ty>::digits + can_negative;
		constexpr auto buff_last = buff_size - 1;

		char_type buff[buff_size];
		char_type* begin = buff + buff_size;
		const bool is_negative = _num < 0;

		if (is_negative) {
			*buff = '1';
		}
		else {
			*buff = '0';
		}

		auto num = db::to_unsigned(_num);

		if constexpr (FULL_OUTPUT) {
			for (db::int32_t i = 0; i != buff_size - can_negative; ++i) {
				*--begin = (num & 1) + '0';
				num /= 2;
			}
		}
		else {
			while (num) {
				*--begin = (num & 1) + '0';
				num /= 2;
			}

		}

		return std::basic_string<char_type>((FULL_OUTPUT == true ? buff : begin), (FULL_OUTPUT == true ? buff_size : (buff + buff_size) - begin));
	}

	// * Number to decimal string
	template <class char_type = char, class Ty = db::int32_t>
	constexpr inline std::basic_string<char_type> ntods(const Ty _num) noexcept
	{
		constexpr static auto digits2 = [](const db::size_t _num) noexcept -> const char* const
		{
			return &"0001020304050607080910111213141516171819"
				"2021222324252627282930313233343536373839"
				"4041424344454647484950515253545556575859"
				"6061626364656667686970717273747576777879"
				"8081828384858687888990919293949596979899"[_num * 2];
		};
		constexpr auto can_negative = !std::is_unsigned_v<Ty>;
		constexpr auto buff_size = std::numeric_limits<Ty>::digits10 + can_negative + 2;
		constexpr auto buff_last = buff_size - 1;

		char_type buff[buff_size];
		char_type* begin = buff + buff_last;
		const bool is_negative = _num < 0;

		auto num = db::to_unsigned(_num);
		db::uint8_t tmp;

		while (num >= 100) {
			tmp = num % 100;
			num /= 100;
			std::memcpy(begin -= 2, digits2(tmp), 2);
		}

		if (num < 10) {
			*--begin = static_cast<char_type>('0' + num);
		}
		else {
			std::memcpy(begin -= 2, digits2(num), 2);
			if (is_negative) {
				*--begin = static_cast<char_type>('-');
			}
		}

		return std::basic_string<char_type>(begin, (buff + buff_last) - begin);
	}

	// * Number to Hex string
	template <class char_type = char, class Ty = db::int32_t>
	constexpr inline std::basic_string<char_type> ntohs(const Ty _num) noexcept
	{
		constexpr auto can_negative = !std::is_unsigned_v<Ty>;
		constexpr auto buff_size = std::numeric_limits<Ty>::digits10 + can_negative;
		constexpr auto buff_last = buff_size - 1;

		char_type buff[buff_size];
		char_type* begin = buff + buff_last;
		const bool is_negative = _num < 0;

		auto num = db::to_unsigned(_num);
		db::uint8_t tmp;

		while (num) {
			tmp = num % 16;
			if (tmp >= 10) {
				*begin-- = static_cast<char_type>(tmp + 'A' - 10);
			}
			else {
				*begin-- = static_cast<char_type>(tmp + '0');
			}
			num /= 16;
		}

		if (is_negative) {
			*begin = static_cast<char_type>('-');
		}
		else {
			++begin;
		}

		return std::basic_string<char_type>(begin, (buff + buff_size) - begin);
	}
}

// STRING TO NUMBER
namespace db
{
	// * Floatingpoint string to double
	template <class char_type>
	constexpr inline double fstod(const char_type* _str)
	{
		const bool is_negative = (*_str == '-' ? ++_str : false);
		const db::size_t _len = db::strlen(_str);

		bool has_point = false;
		db::uint8_t current;
		std::double_t num = 0.0;
		db::size_t len = _len;

		while (static_cast<db::uint32_t>(is_negative) != len--) {
			current = _str[len];

			if (current == '.') {
				--len;
				has_point = true;
				break;
			}
			else {
				num += (current - '0');
				num /= 10;
			}
		}

		if (has_point) {
			db::size_t before = 0;

			for (db::size_t i = 0; i != len; ++i) {
				current = _str[i];
				before += (current - '0');
				before *= 10;
			}

			current = _str[len];
			before += (current - '0');

			num += before;
		}
		else {
			num = static_cast<db::size_t>(num * std::pow(10, _len));
		}

		return num;
	}

	// * Binary string to number
	template <class Ty = db::size_t, class char_type>
	constexpr inline Ty bston(const char_type* const _str, const bool _big_endian = true)
	{
		constexpr auto can_negative = !std::is_unsigned_v<Ty>;
		constexpr auto num_max = std::numeric_limits<Ty>::max();
		const db::size_t _len = db::strlen(_str);

		const bool is_negative = (_big_endian == true ? (*_str == '1') : *(_str + _len) == '1');

		Ty num = 0;
		Ty tmp_num = 0;

		for (db::size_t i = 0; i < _len; ++i)
		{
			if (_str[i] != '0' && _str[i] != '1') {
				throw std::invalid_argument("Only binary strings");
			}

			if (_str[i] == '1') {
				const db::size_t _expo = (_big_endian == true ? (_len - i - 1) : i);
				if (_expo == 0) {
					tmp_num += 1;
				}
				else {
					tmp_num += static_cast<db::size_t>(2) << (_expo - 1);
				}
			}
			else {
				continue;
			}

			if (tmp_num < num) {
				return num_max;
			}

			num = tmp_num;
		}

		if (can_negative) {
			if (is_negative) {
				num *= static_cast<Ty>(-1);
			}
		}

		return num;
	}

	// * Decimal string to number
	template <class Ty = db::size_t, class char_type>
	constexpr inline Ty dston(const char_type* _str)
	{
		constexpr auto can_negative = !std::is_unsigned_v<Ty>;
		constexpr auto num_max = std::numeric_limits<Ty>::max();

		const bool is_negative = (*_str == '-' ? ++_str : false);

		Ty num = 0;
		Ty tmp_num = 0;

		while (*_str)
		{
			if (*_str < '0' || *_str > '9') {
				throw std::invalid_argument("only number strings");
			}

			tmp_num = (num * 10) + (*_str++ - '0');

			if (tmp_num < num) {
				return num_max;
			}

			num = tmp_num;
		}

		if (is_negative) {
			num *= static_cast<Ty>(-1);
		}

		return num;
	}

	// * Hex string to number
	template <class Ty = db::size_t, class char_type>
	constexpr inline Ty hston(const char_type* _str)
	{
		constexpr auto can_negative = !std::is_unsigned_v<Ty>;
		constexpr auto num_max = std::numeric_limits<Ty>::max();

		const bool is_negative = (*_str == '-' ? ++_str : false);

		if (*_str == 'x' || *_str == 'X') {
			_str += 2;
		}

		Ty num = 0;
		Ty tmp_num = 0;

		while (*_str)
		{
			tmp_num = num * 16;

			if (*_str >= '0' && *_str <= '9') {
				tmp_num += *_str++ - '0';
			}
			else if (*_str >= 'a' && *_str <= 'f') {
				tmp_num += 10 + *_str++ - 'a';
			}
			else if (*_str >= 'A' && *_str <= 'F') {
				tmp_num += 10 + *_str++ - 'A';
			}
			else {
				throw std::invalid_argument("only hex number strings");
			}

			if (tmp_num < num) {
				return num_max;
			}

			num = tmp_num;
		}

		if constexpr (can_negative) {
			if (is_negative) {
				num *= static_cast<Ty>(-1);
			}
		}

		return num;
	}

	// * Number string to number
	// Binary
	// Decimal
	// Hexadecimal
	// - not 100% accurate
	template <class Ty = db::size_t, class char_type>
	constexpr inline Ty ston(const char_type* _str, const bool _big_endian_if_bin = true)
	{
		switch(db::guess_base<char_type>(_str))
		{
		case 2:
			return db::bston<Ty, char_type>(_str, _big_endian_if_bin);
		case 10:
			return db::dston<Ty, char_type>(_str);
		case 16:
			return db::hston<Ty, char_type>(_str);
		default:
			return 0;
		}
	}
}

// SEARCH
namespace db
{
	// * Binary search
	template<class fIter, class Ty, class Fn>
	constexpr inline fIter binary_search(fIter _first, fIter _last, const Ty& _what, Fn&& _cmp_func)
	{
		fIter iter;
		std::ptrdiff_t count = std::distance(_first, _last);
		db::size_t step;

		while (count > 0LL) {
			step = count / 2ULL;
			iter = _first;
			std::advance(iter, step);

			if (_cmp_func(*iter, _what)) {
				_first = ++iter;
				count -= step + 1ULL;
			}
			else {
				count = step;
			}
		}

		return _first;
	}

	// * Binary search
	template<class fIter, class Ty, class Fn>
	constexpr inline fIter binary_search(fIter _first, fIter _last, Ty&& _what, Fn&& _cmp_func)
	{
		db::binary_search<fIter, Ty>(_first, _last, _what, std::move(_cmp_func));
	}

#ifdef _XSTDDEF_
	// * Binary search ( std::less )
	template<class fIter, class Ty>
	constexpr inline fIter binary_search(fIter _first, fIter _last, const Ty& _what)
	{
		return db::binary_search<fIter, Ty>(_first, _last, _what, std::move(std::less<>{}));
	}

	// * Binary search ( std::less )
	template<class fIter, class Ty>
	constexpr inline fIter binary_search(fIter _first, fIter _last, Ty&& _what)
	{
		return db::binary_search<fIter, Ty>(_first, _last, _what, std::move(std::less<>{}));
	}
#endif
}

// OUTPUT
namespace db
{
#ifdef _IOSTREAM_
	// * Print vector
	template <class ostream = std::ostream, class VecTy>
	inline void print_vec(const std::vector<VecTy>& _vec, const char _delim = ',') noexcept
	{
#if _HAS_CXX17
		if constexpr ((std::is_same_v<ostream, std::ostream>)) {
#else
		if ((std::is_same_v<ostream, std::ostream>)) {
#endif
			for (db::size_t i = 0; i != _vec.size() - 1; ++i) {
				std::cout << _vec.at(i) << _delim << ' ';
			}
			std::cout << _vec.back();
		}
		else {
			for (db::size_t i = 0; i != _vec.size() - 1; ++i) {
				std::wcout << _vec.at(i) << _delim << ' ';
			}
			std::wcout << _vec.back();
		}
		}

	// * Print vector
	template <class ostream = std::ostream, class VecTy>
	inline void print_vec(std::vector<VecTy> && _vec, const char _delim = ',') noexcept
	{
		db::print_vec(_vec, _delim);
	}
#endif
	// * Lenght of printf output
	template <db::size_t _Buffer = 1024, class... Args>
	inline db::size_t scprintf(const char* _format, const Args&... _args) noexcept
	{
		char _buff[_Buffer + 1]{ char() };
		snprintf(_buff, _Buffer + 1, _format, _args...);
		return strnlen(_buff, _Buffer + 1);
	}

	// * Lenght of printf output
	template <db::size_t _Buffer = 1024, class... Args>
	inline db::size_t scprintf(const char* _format, Args&&... _args) noexcept
	{
		return db::scprintf(_format, _args...);
	}
}
