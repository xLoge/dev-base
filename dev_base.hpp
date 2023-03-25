#include <stdint.h>
#include <vector>
#include <string>
#include <immintrin.h>
#include <algorithm>
#include <utility>
#include <iterator>

// MEM OPERATIONS
namespace db
{
	template <class Ty>
	constexpr inline Ty* raw_memcpy(Ty* _dst, const Ty* const _src, std::size_t _size) noexcept
	{
		for (std::size_t i = 0; i != _size; ++i) {
			_dst[i] = _src[i];
		}
		return _dst;
	}

	// * constexpr memcpy for any type of char
	template <class Ty>
	constexpr inline Ty* memcpy(Ty* _dst, const Ty* const _src, std::size_t _size) noexcept
	{
		if (std::is_constant_evaluated()) {
			return db::raw_memcpy<Ty>(_dst, _src, _size);
		}
		else {
			return static_cast<Ty*>(std::memcpy(_dst, _src, _size * sizeof(Ty)));
		}
	}

	template <class Ty>
	constexpr inline std::int32_t raw_memcmp(const Ty* const _buff1, const Ty* const _buff2, std::size_t _size) noexcept
	{
		for (; 0 < _size; ++_buff1, ++_buff2, --_size) {
			if (*_buff1 != *_buff2) {
				return *_buff1 < *_buff2 ? -1 : 1;
			}
		}
		return 0;
	}

	template <class Ty>
	constexpr inline std::int32_t memcmp(const Ty* const _buff1, const Ty* const _buff2, std::size_t _size) noexcept
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
	constexpr inline const Ty* rmemchr(const Ty* _str, const Ty _ch, std::size_t _len)
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

	template <class Ty>
	constexpr inline const Ty* raw_memchr(const Ty* _str, const Ty _ch, const std::size_t _maxlen) noexcept
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
	constexpr inline const Ty* memchr(const Ty* _str, const Ty _ch, const std::size_t _maxlen) noexcept
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
	// * constexpr strlen base for any char type
	template <class char_type>
	constexpr inline std::size_t raw_strlen(const char_type* _str) noexcept
	{
		const char_type* _begin = _str;
		for (; *_begin; ++_begin);
		return (_begin - _str);
	}

	// * constexpr strlen for any type of char
	template <class char_type>
	constexpr inline std::size_t strlen(const char_type* _str) noexcept
	{
		if (std::is_constant_evaluated()) {
			return db::raw_strlen<char_type>(_str);
		}
		else {
			if constexpr (
				std::is_same_v<char, char_type> 
#ifdef				__cpp_char8_t
				|| std::is_same_v<char8_t, char_type>
#endif
				)
			{
				return std::strlen(reinterpret_cast<const char*>(_str));
			}
			else if constexpr (
				std::is_same_v<wchar_t, char_type> ||
				std::is_same_v<char16_t, char_type>
				)
			{
				return std::wcslen(reinterpret_cast<const wchar_t*>(_str));
			}
			else if constexpr (
				std::is_same_v<char32_t, char_type>	
				)
			{
				return db::raw_strlen<char_type>(reinterpret_cast<const char32_t*>(_str));
			}
			else
			{
				throw std::invalid_argument("no type of char specified.");
			}
		}
	}

	// * fast strlen version I wrote 
	inline size_t fast_strlen(const char* _str)
	{
		static auto is_any_zero_32 = [](const void* data) noexcept -> bool {
			const __m256i zero256 = _mm256_set1_epi8(0);
			const __m256i data_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data));
			const int32_t mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(data_vec, zero256));
			return mask != 0;
		};

		static auto is_any_zero_16 = [](const void* data) noexcept -> bool {
			const __m128i zero128 = _mm_set1_epi8(0);
			const __m128i data_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data));
			const int32_t mask = _mm_movemask_epi8(_mm_cmpeq_epi8(data_vec, zero128));
			return mask != 0;
		};

		const char* _current = _str;

		// skip big chunks
		while (is_any_zero_32(_current) == false) {
			_current += 32U;
		}

		// skip last mid chunk
		if (is_any_zero_16(_current) == false) {
			_current += 16U;
		}

		// count rest one by one
		for (; *_current != char(); ++_current);

		return static_cast<std::size_t>(_current - _str);
	}

	// * guesses base of numeric string
	template <class char_type>
	constexpr inline std::uint16_t guess_base(const char_type* _str) noexcept
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
	template <class To = std::int64_t, class From>
	constexpr inline To to_signed(const From _num) noexcept
	{
		constexpr bool is_unsigned = std::is_unsigned_v<From>;

#if _HAS_CXX17
		if constexpr (is_unsigned) {
#else 
		if (is_unsigned) {
#endif
			return static_cast<To>(_num);
		}

		return _num;
		}

	// * inverse square root function
	inline float inv_sqrt(const float _val)
	{
		__m128 invsqrt = _mm_set_ss(_val);
		invsqrt = _mm_rsqrt_ss(invsqrt);
		return _mm_cvtss_f32(invsqrt);
	}
}

// NUMBER TO STRING
namespace db
{
	// * Floatingpoint to string
	template <std::size_t _PRECISION = 5, class char_type = char, class Ty = float>
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
		std::size_t first = static_cast<std::size_t>(_num);
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

			std::size_t maxlen = _PRECISION;

			while (_num && maxlen--)
			{
				first = static_cast<std::size_t>(_num);
				_num -= first;
				_num *= 10.0;
				*begin++ = static_cast<char_type>(first + '0');
			}
		}

		return std::basic_string<char_type>(buff, static_cast<std::size_t>(begin - buff));
	}

	// * Number to binary string
	template <class char_type = char, class Ty = std::int32_t, bool FULL_OUTPUT = true>
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
			for (std::int32_t i = 0; i != buff_size - can_negative; ++i) {
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
	template <class char_type = char, class Ty = std::int32_t>
	constexpr inline std::basic_string<char_type> ntods(const Ty _num) noexcept
	{
		constexpr static auto digits2 = [](const std::size_t _num) noexcept -> const char* const
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
		std::uint8_t tmp;

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
	template <class char_type = char, class Ty = std::int32_t>
	constexpr inline std::basic_string<char_type> ntohs(const Ty _num) noexcept
	{
		constexpr auto can_negative = !std::is_unsigned_v<Ty>;
		constexpr auto buff_size = std::numeric_limits<Ty>::digits10 + can_negative;
		constexpr auto buff_last = buff_size - 1;

		char_type buff[buff_size];
		char_type* begin = buff + buff_last;
		const bool is_negative = _num < 0;

		auto num = db::to_unsigned(_num);
		std::uint8_t tmp;

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
		const std::size_t _len = db::strlen(_str);

		bool has_point = false;
		std::uint8_t current;
		std::double_t num = 0.0;
		std::size_t len = _len;

		while (static_cast<std::uint32_t>(is_negative) != len--) {
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
			std::size_t before = 0;

			for (std::size_t i = 0; i != len; ++i) {
				current = _str[i];
				before += (current - '0');
				before *= 10;
			}

			current = _str[len];
			before += (current - '0');

			num += before;
		}
		else {
			num = static_cast<std::size_t>(num * std::pow(10, _len));
		}

		return num;
	}

	// * Binary string to number
	template <class Ty = std::size_t>
	constexpr inline Ty bston(const char* const _str, const bool _big_endian = true)
	{
		constexpr auto can_negative = !std::is_unsigned_v<Ty>;
		constexpr auto num_max = std::numeric_limits<Ty>::max();
		const std::size_t _len = db::strlen(_str);

		const bool is_negative = (_big_endian == true ? (*_str == '1') : *(_str + _len) == '1');

		Ty num = 0;
		Ty tmp_num = 0;

		for (std::size_t i = 0; i < _len; ++i)
		{
			if (_str[i] != '0' && _str[i] != '1') {
				throw std::invalid_argument("Only binary strings");
			}

			if (_str[i] == '1') {
				const auto _expo = (_big_endian == true ? (_len - i - 1) : i);
				if (_expo == 0) {
					tmp_num += 1;
				}
				else {
					tmp_num += static_cast<std::size_t>(2) << (_expo - 1);
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
	template <class Ty = std::size_t>
	constexpr inline Ty dston(const char* _str)
	{
		constexpr auto can_negative = !std::is_unsigned_v<Ty>;
		constexpr auto num_max = std::numeric_limits<Ty>::max();

		const bool is_negative = (*_str == '-' ? ++_str : false);

		Ty num = 0;

		while (*_str)
		{
			if (*_str < '0' || *_str > '9') {
				throw std::invalid_argument("only number strings");
			}

			const Ty tmp_num = (num * 10) + (*_str++ - '0');

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
	template <class Ty = std::size_t>
	constexpr inline Ty hston(const char* _str)
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
	template <class Ty = std::size_t>
	constexpr inline Ty ston(const char* _str, const bool _big_endian_if_bin = true)
	{
		switch (guess_base(_str))
		{
		case 2:
			return db::bston<Ty>(_str, _big_endian_if_bin);
		case 10:
			return db::dston<Ty>(_str);
		case 16:
			return db::hston<Ty>(_str);
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
		std::size_t step;

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
}

// OUTPUT
namespace db
{
	// * Print vector
	template <class ostream = std::ostream, class VecTy>
	inline void print_vec(const std::vector<VecTy>& _vec, const char _delim = ',') noexcept
	{
#if _HAS_CXX17
		if constexpr ((std::is_same_v<ostream, std::ostream>)) {
#else
		if ((std::is_same_v<ostream, std::ostream>)) {
#endif
			for (std::size_t i = 0; i != _vec.size() - 1; ++i) {
				std::cout << _vec.at(i) << _delim << ' ';
			}
			std::cout << _vec.back();
		}
		else {
			for (std::size_t i = 0; i != _vec.size() - 1; ++i) {
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

	// * Lenght of printf output
	template <std::size_t _Buffer = 1024, class... Args>
	inline std::size_t scprintf(const char* _format, const Args&... _args) noexcept
	{
		char _buff[_Buffer + 1]{ char() };
		snprintf(_buff, _Buffer + 1, _format, _args...);
		return strnlen(_buff, _Buffer + 1);
	}

	// * Lenght of printf output
	template <std::size_t _Buffer = 1024, class... Args>
	inline std::size_t scprintf(const char* _format, Args&&... _args) noexcept
	{
		return db::scprintf(_format, _args...);
	}
}
