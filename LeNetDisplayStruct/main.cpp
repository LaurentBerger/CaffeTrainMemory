#include <caffe/caffe.hpp>
#include <memory>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/ocl.hpp>
#include <fstream>
#include "caffe/layers/memory_data_layer.hpp"

using namespace std;
using namespace cv;

template<class type>
void Lsb2Msb(type &a)
{
    char *x = (char *)&a;
    int nb = sizeof(type);
    for (int i=0;i<nb/2;i++)
        swap(x[i], x[nb-i-1]);
}

void MnistToMat(char *nameData, char *nameLabel, vector<Mat> &trainImages, std::vector<uint> &labels, bool display = false);
void VectorMat2VectorFloat(vector<Mat> img, vector<uint> lab, vector<float> &data, vector<float> &label);

const String keys =
"{Aide h usage ? help  |     | Afficher ce message   }"
"{solver or model prototxt s    |lenet_solverleger.prototxt     | solver param }"
"{train model t     |1     | train model }"
"{ m model name    | lenet_Shape.prototxt    | model name }"
"{ b binary name    |lenet_train_Shape.binary     | binary name }"
;

int main(int argc, char **argv)
{
    cv::ocl::setUseOpenCL(false);
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
    Mat img = imread("g:/lib/opencv/samples/data/fruits.jpg", IMREAD_COLOR);
    cout << img.at<Vec3b>(0, 0);
    imwrite("Fruitsopencv.png", img);


    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    String modelTxt = parser.get<String>("m");;
    String modelBin = parser.get<String>("b");;


    caffe::Caffe::set_mode(caffe::Caffe::CPU);
    caffe::NetParameter netCaffe;
    caffe::NetParameter netBinary;

    boost::shared_ptr<caffe::Net<float> > reseau;
    reseau.reset(new caffe::Net<float>(modelTxt, caffe::TEST));
    reseau->CopyTrainedLayersFrom(modelBin);


    
    for (int i = 0; i < 10; i++)
    {
        caffe::Blob<float> *input_layer = reseau->input_blobs()[0];
        int numChannels = input_layer->channels();
        cv::Size inputGeometry = cv::Size(input_layer->width(), input_layer->height());
        input_layer->Reshape(1, numChannels,inputGeometry.height, inputGeometry.width);
        Mat digit(28, 28, CV_8UC1, Scalar(0));
        putText(digit, format("%d",i), Point(3, 23), FONT_HERSHEY_COMPLEX, 0.9, Scalar(255), 2);
        imshow("DIGIT", digit);
        waitKey(50);
  
        Mat digitFloat;
        digit.convertTo(digitFloat, CV_32F, 1 / 256.0,-0.1);
        float* input_data = input_layer->mutable_cpu_data();
        memcpy(input_data,digitFloat.data,  28 * 28 * 4);
        
        reseau->Forward();
        /* Copy the output layer to a std::vector */
        caffe::Blob<float>* output_layer = reseau->output_blobs()[0];
        const float* begin = output_layer->cpu_data();
        const float* end = begin + output_layer->channels();
        for (auto &item = begin; item != end; item++)
            cout << *item << " ";
        cout << "\n";
    }
    vector<string> nomCouche;
    vector<string> nomBlob;

    nomCouche = reseau->layer_names();
    nomBlob = reseau->blob_names();
    FileStorage fs;
    fs.open("weights.yml", FileStorage::WRITE);
    string searchName("conv1");
    vector<Mat> selectFilter;
    for (auto &nom : nomCouche)
    {
        const boost::shared_ptr<caffe::Layer<float> > &conv_layer= reseau->layer_by_name(nom);
        fs <<  nom << "[";
        fs << "{";
        fs << "Type" << conv_layer->type();
        if (conv_layer->blobs().size() > 0)
        {
            boost::shared_ptr<caffe::Blob<float> >& weight = conv_layer->blobs()[0];
            float* conv_weight = weight->mutable_cpu_data();
            if (fs.isOpened())
            {
                vector<int> s = weight->shape();
                fs << "WeightsShape" << s;
                switch(s.size())
                {
                case 1:
                {
                    fs << "M";
                    Mat a(s[0], 1, CV_32FC1, conv_weight );
                    fs << a;
                    break;

                }
                case 2 :
                {
                    fs << "M";
                    Mat a(s[0], s[1], CV_32FC1, conv_weight );
                    fs << a;
                }
                    break;
                case 3:
                    for (int i = 0; i < s[0]; i++)
                    {
                        fs << format("M%d", i);
                        Mat a(s[1], s[2], CV_32FC1, conv_weight + s[2] * s[3] * i );
                        fs << a;
                    }
                    break;
                case 4:
             
                    for (int i = 0; i < s[0]; i++)
                    {
                        for (int j = 0; j < s[1]; j++)
                        {
                            fs << format("M%dX%d", i, j);
                            Mat a(s[2], s[3], CV_32FC1, conv_weight + s[2] * s[3] * s[1] * i + s[2] * s[3] * j);
                            fs << a;
                            if (nom == searchName)
                                selectFilter.push_back(a.clone());
                        }
                    }
                    break;

                }

            }
            if (conv_layer->blobs().size() > 1) 
            {
                boost::shared_ptr<caffe::Blob<float> >& bias = conv_layer->blobs()[1];
                float* conv_bias = bias->mutable_cpu_data();
                if (fs.isOpened())
                {
                    vector<int> s = bias->shape();
                    fs << "BiasShape" << s;
                    switch (s.size())
                    {
                    case 1:
                    {
                        fs << "B";
                        Mat a(s[0], 1, CV_32FC1, conv_bias);
                        fs << a;
                        break;

                    }
                    case 2:
                    {
                        fs << "B";
                        Mat a(s[0], s[1], CV_32FC1, conv_bias);
                        fs << a;
                    }
                    break;
                    case 3:
                        for (int i = 0; i < s[0]; i++)
                        {
                            fs << format("B%d", i);
                            Mat a(s[1], s[2], CV_32FC1, conv_bias + s[2] * s[3] * i);
                            fs << a;
                        }
                        break;
                    case 4:
                        for (int i = 0; i < s[0]; i++)
                        {
                            for (int j = 0; j < s[1]; j++)
                            {
                                fs << format("B%dX%d", i, j);
                                Mat a(s[2], s[3], CV_32FC1, conv_bias + s[2] * s[3] * s[1] * i + s[2] * s[3] * j);
                                fs << a;
                            }
                        }
                        break;

                    }

                }

            }

        }
        fs << "}";
        fs << "]";

    }

    if (selectFilter.size() != 0)
    {
        int ind = 0;
        for (auto &img : selectFilter)
        {
            Mat x(256, 256, CV_32FC1,Scalar(0));
            img.copyTo(x(Rect(Point(0, 0), Size(img.cols, img.rows))));
            FileStorage fs(format("img%d.yml", ind++),FileStorage::WRITE);
            fs << "Image" << x;
            fs.release();
        }
    }
    dnn::Net net;
    try {
        net = dnn::readNetFromCaffe(modelTxt, modelBin);
    }
    catch (const cv::Exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        if (net.empty())
        {
            std::cerr << "Can't load network by using the following files: " << std::endl;
            std::cerr << "prototxt:   " << modelTxt << std::endl;
            std::cerr << "caffemodel: " << modelBin << std::endl;
            exit(-1);
        }
    }

    int ind = 4;
    Ptr<dnn::Layer> l = net.getLayer("conv1");
    vector<Mat> bb = l->blobs;
    return 0;
}

void VectorMat2VectorFloat(vector<Mat> img, vector<uint> lab, vector<float> &data, vector<float> &label) 
{
    data.resize(img.size()*img[0].total());
    label.resize(img.size());


    for (int i = 0; i<img.size(); ++i)
    {
        Mat imF;
        img[i].convertTo(imF, CV_32F,1./256,-0.1);
        memcpy(&data[i*img[0].total()], imF.data, imF.total()*4);
        label[i] = lab[i];
    }

}


void MnistToMat(char *nameData,char *nameLabel,vector<Mat> &trainImages, std::vector<uint> &labels, bool display)
{
    ifstream fMnistTrain, fMnistLabel;

    fMnistTrain.open(nameData, ios::binary + ios::in);
    if (!fMnistTrain.is_open())
    {
        cout << "File not found\n";
        return ;
    }
    fMnistLabel.open(nameLabel, ios::binary + ios::in);
    if (!fMnistLabel.is_open())
    {
        cout << "File not found\n";
        return ;
    }

    int magicNumber, nbItems, nbRows, nbCols;
    fMnistTrain.read((char*)&magicNumber, 4);
    fMnistTrain.read((char*)&nbItems, 4);
    fMnistTrain.read((char*)&nbRows, 4);
    fMnistTrain.read((char*)&nbCols, 4);
    Lsb2Msb(nbItems);
    Lsb2Msb(nbRows);
    Lsb2Msb(nbCols);
    std::vector<cv::Mat> ;
    for (int i = 0; i < nbItems; i++)
    {
        cv::Mat x(nbRows, nbCols, CV_8UC1);
        fMnistTrain.read((char*)x.data, nbRows* nbCols);
        trainImages.push_back(x);
    }
    int nbLabels;
    fMnistLabel.read((char*)&magicNumber, 4);
    fMnistLabel.read((char*)&nbLabels, 4);
    Lsb2Msb(nbLabels);

    for (int i = 0; i < nbItems; i++)
    {
        uchar c;
        fMnistLabel.read((char*)&c, 1);
        labels.push_back(c);
        if (i % 1000 == 0 && display)
        {
            imshow(format("Img %d", c), trainImages[i]);
            waitKey(10);
        }
    }

}