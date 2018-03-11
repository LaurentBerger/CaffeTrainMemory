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


const String keys =
"{Aide h usage ? help  |     | Afficher ce message   }"
"{model m prototxt     |lenet_train_leger     |model }"
"{weights b binary     |lenet_train_leger     |binary }"
;

int main(int argc, char **argv)
{
    cv::ocl::setUseOpenCL(false);
    caffe::Caffe::set_mode(caffe::Caffe::CPU);


    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    vector<vector<vector<Mat>>> netWeightsPerLayer;

    for (int i = 6; i <= 10; i++)
    {
        string modelTxt = parser.get<string>("m");
        string modelBin = parser.get<String>("b");
        modelTxt = modelTxt+to_string(i)+".prototxt";
        modelBin = modelBin+to_string(i)+".prototxt.binary";
        caffe::Caffe::set_mode(caffe::Caffe::CPU);
        caffe::NetParameter netCaffe;
        caffe::NetParameter netBinary;

        boost::shared_ptr<caffe::Net<float> > reseau;
        reseau.reset(new caffe::Net<float>(modelTxt, caffe::TEST));
        reseau->CopyTrainedLayersFrom(modelBin);



        vector<string> nomCouche;
        vector<string> nomBlob;

        nomCouche = reseau->layer_names();
        nomBlob = reseau->blob_names();
        vector<vector<Mat>> weightsPerLayer;
        for (auto &nom : nomCouche)
        {
            vector<Mat> weightsLayer;
            vector<Mat> biasLayer;
            const boost::shared_ptr<caffe::Layer<float> > &conv_layer = reseau->layer_by_name(nom);
            if (conv_layer->blobs().size() > 0)
            {
                boost::shared_ptr<caffe::Blob<float> >& weight = conv_layer->blobs()[0];
                float* conv_weight = weight->mutable_cpu_data();
                vector<int> s = weight->shape();
                switch (s.size())
                {
                case 1:
                {
                    Mat a(s[0], 1, CV_32FC1, conv_weight);
                    weightsLayer.push_back(a.clone());
                    break;

                }
                case 2:
                {
                    Mat a(s[0], s[1], CV_32FC1, conv_weight);
                    weightsLayer.push_back(a.clone());
                }
                break;
                case 3:
                    for (int i = 0; i < s[0]; i++)
                    {
                        Mat a(s[1], s[2], CV_32FC1, conv_weight + s[2] * s[3] * i);
                        weightsLayer.push_back(a.clone());
                    }
                    break;
                case 4:

                    for (int i = 0; i < s[0]; i++)
                    {
                        for (int j = 0; j < s[1]; j++)
                        {
                            Mat a(s[2], s[3], CV_32FC1, conv_weight + s[2] * s[3] * s[1] * i + s[2] * s[3] * j);
                            weightsLayer.push_back(a.clone());
                        }
                    }
                    break;

                }

            }
            if (conv_layer->blobs().size() > 1)
            {
                boost::shared_ptr<caffe::Blob<float> >& bias = conv_layer->blobs()[1];
                float* conv_bias = bias->mutable_cpu_data();
                vector<int> s = bias->shape();
                switch (s.size())
                {
                case 1:
                {
                    Mat a(s[0], 1, CV_32FC1, conv_bias);
                    biasLayer.push_back(a.clone());
                    break;

                }
                case 2:
                {
                    Mat a(s[0], s[1], CV_32FC1, conv_bias);
                    biasLayer.push_back(a.clone());
                }
                break;
                case 3:
                    for (int i = 0; i < s[0]; i++)
                    {
                        Mat a(s[1], s[2], CV_32FC1, conv_bias + s[2] * s[3] * i);
                        biasLayer.push_back(a.clone());
                    }
                    break;
                case 4:
                    for (int i = 0; i < s[0]; i++)
                    {
                        for (int j = 0; j < s[1]; j++)
                        {
                            Mat a(s[2], s[3], CV_32FC1, conv_bias + s[2] * s[3] * s[1] * i + s[2] * s[3] * j);
                            biasLayer.push_back(a.clone());
                        }
                    }
                    break;

                }
            weightsPerLayer.push_back(weightsLayer);
            weightsPerLayer.push_back(biasLayer);
            weightsLayer.clear();
            biasLayer.clear();
            }
        }
        netWeightsPerLayer.push_back(weightsPerLayer);
        weightsPerLayer.clear();
    }
    vector<vector<double>> distance;
    for (int i = 0; i < netWeightsPerLayer.size(); i++)
    {
        for (int j = i + 1; j < netWeightsPerLayer.size(); j++)
        {
            cout << "Net " << i << " With Net " << j << endl;
            vector<double> d;
            for (int indCouche=0; indCouche<1; indCouche++)
                for (int indPoidsI = 0; indPoidsI < netWeightsPerLayer[i][indCouche].size(); indPoidsI++)
                {
                    for (int indPoidsJ = 0; indPoidsJ < netWeightsPerLayer[i][indCouche].size(); indPoidsJ++)
                    {
                        Mat dst;
                        absdiff(netWeightsPerLayer[i][indCouche][indPoidsI], netWeightsPerLayer[j][indCouche][indPoidsJ],dst);
                        double x =sum(dst)[0];
                        cout << x << "\t";
                    }
                    cout << "\n";

                }

        }
    }
    return 0;
}

