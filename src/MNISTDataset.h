#ifndef MNISTLOADER_H_
#define MNISTLOADER_H_

#include <vector>
#include <opencv2/core/core.hpp>

template<typename T>
class MNISTDataset {
public:
	enum IdxDatatype {
		UBYTE = 0x08,
		SBYTE = 0x09,
		SHORT = 0x0B,
		INT = 0x0C,
		FLOAT = 0x0D,
		DOUBLE = 0x0E,
	};

	MNISTDataset(std::string const& file);
	virtual ~MNISTDataset();

	/**
	 * Loads the dataset.
	 * @return False in case of an error.
	 */
	bool load();

	IdxDatatype getDatatype() const;
	size_t size() const;
	std::string getPath() const;
	T& operator[](int const& idx);
	T const& operator[](int const& idx) const;

	typedef typename std::vector<T>::iterator iterator;

	iterator begin();
	iterator end();

private:
	std::string const m_FileName;
	/** Number of images in the dataset. */
	size_t m_Count;
	IdxDatatype m_Datatype;
	std::vector<T> m_Images;
};

typedef MNISTDataset<cv::Mat> MNISTImageDataset;
typedef MNISTDataset<uint8_t> MNISTLableDataset;

#endif /* MNISTLOADER_H_ */
