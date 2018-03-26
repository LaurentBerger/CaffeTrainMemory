#include <caffe/caffe.hpp>
#include <memory>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/ocl.hpp>
#include <fstream>

#include "caffe/layers/memory_data_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

using namespace std;
using namespace cv;

void main(void)
{
    vector<String> fileNames;
    // Structure N filtres par itération , M itération par fichiers, P réseaux filters[p][m][n]
    vector<vector<vector<Mat>>> filters;

    FileStorage f;
    f.open("test.yml", FileStorage::APPEND);
    f << "a" << 28.54;
    f.release();
    f.open("test.yml", FileStorage::READ);
    cout << " First read "<<float(f["a"]) << endl;
    f.release();
    f.open("test.yml", FileStorage::APPEND + FileStorage::WRITE);
    f << "b" << 323.54;
    f.release();
    f.open("test.yml", FileStorage::APPEND + FileStorage::WRITE);
    cout << " second read "<< float(f["a"]) << endl;
    cout << " second read " << float(f["b"]) << endl;
    f.release();


    glob("conv1o*.yml", fileNames);
    int nbIterRef = -1;
    int nbFilters = -1;
    for (auto file : fileNames)
    {
        FileStorage fs;
        fs.open(file, FileStorage::READ);
        FileNode n = fs.root();
        FileNodeIterator it = n.begin();
        for (; it != n.end(); it++)
        {
            //cout << it.readRaw( << endl;
        }


    }


}
