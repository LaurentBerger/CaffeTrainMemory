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

    shared_ptr<caffe::Net<float> > net_;
    net_.reset(new caffe::Net<float>(modelTxt, caffe::TEST));
    net_->CopyTrainedLayersFrom(modelBin);


    
    for (int i = 0; i < 10; i++)
    {
        caffe::Blob<float> *input_layer = net_->input_blobs()[0];
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
        
        net_->Forward();

        /* Copy the output layer to a std::vector */
        caffe::Blob<float>* output_layer = net_->output_blobs()[0];
        const float* begin = output_layer->cpu_data();
        const float* end = begin + output_layer->channels();
        for (auto &item = begin; item != end; item++)
            cout << *item << " ";
        cout << "\n";
    }

//    caffe::Blob<float>* input_layer = net_->input_blobs()[0];


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