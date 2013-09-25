#if !defined( BACKPROPAGATION_HPP )
#define BACKPROPAGATION_HPP

#include <cstdio>
#include <cstdlib>
#include <cmath>

#define for if ( 0 ) ; else for

/**
 * �j���[�����l�b�g���[�N
 * �덷�t�`�d�@(�o�b�N�v���p�Q�[�V����)
 */
class BackPropagation
{
public:
	BackPropagation( int input, int hidden, int output );
	~BackPropagation();

	//! ���͂�^���Č��ʂ𓾂�
	void foward( double* inputLayer );

	//! ���t�f�[�^��^���Ċw�K
	void back( double* inputLayer, double* teach );

	//! �덷�𓾂�
	double getError( double* teach );

	void debugOutput();

private:
	double eta_; //!< �w�K�W��
	double alpha_; //!< �O��̕ω��ʂ���w�K�ʂ��R���g���[�����邽�߂̂���

	int input_; //!< ���͑w��
	int hidden_; //!< ���ԑw��
	int output_; //!< �o�͑w��

	double* hiddenLayer_; //!< ���ԑw
	double* outputLayer_; //!< �o�͑w
	double** wij_; //!< �׏d�i���͑w->���ԑw�j
	double** wjk_; //!< �׏d�i���ԑw->�o�͑w�j

	double* deltaj_; //!< �덷�i���͑w->���ԑw�j
	double* deltak_; //!< �덷�i���ԑw->�o�͑w�j
	double** deltaWji_; //!< �׏d�C���l�i���͑w->���ԑw�j
	double** deltaWkj_; //!< �׏d�C���l�i���͑w->���ԑw�j

private:
	//! �V�O���C�h�֐�
	inline static double sigmoid( double x )
	{
		return 1.0 / ( 1.0 + exp( -x ) );
	}
};

#endif // !defined( BACKPROPAGATION_HPP )
