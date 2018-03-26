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

    glob("conv1o*.yml", fileNames);
    int nbIterRef = -1;
    int nbFilters = -1;
    for (auto file : fileNames)
    {
        cout << "File : " << file << endl;
        FileStorage fs;
        fs.open(file, FileStorage::READ);
        
        int idxStream = 0;
        FileNode n;
        vector<vector<Mat>> w;
        do
        {
            n= fs.root(idxStream++);
            if (!n.empty())
            {
                cout << n.name() <<" "<< n.type()<< endl;
                FileNodeIterator it = n.begin();
                vector<Mat> x;
                for (; it != n.end(); it++)
                {
                    cout << (*it).name() << "\n";
                    fs[(*it).name()]>>x;
                }
                w.push_back(x);
            }
        } 
        while (!n.empty());
        filters.push_back(w);

    }
    for (int r = 0; r < filters.size(); r++)
    {

        for (int ite = 0; ite < filters[r].size(); ite++)
        {
            for (int idxFilter = 0; idxFilter < filters[r][ite].size(); idxFilter++)
            {
                double x = mean(filters[r][ite][idxFilter])[0];
                double m = -1;
                int idx = -1;
                for (int k = 0; k < filters[i].size(); k++)
                {
                    double y = mean(filters[i][k])[0];
                    Mat p;
                    multiply(filters[i - 1][j] - x, filters[i][k] - y, p);
                    y = sum(p)[0];
                    if (y > m)
                    {
                        m = y;
                        idx = k;
                    }
                }
                cout << idx << "\t";

            }
        }
        cout << "\n";
    }

}
