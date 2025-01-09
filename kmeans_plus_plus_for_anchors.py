import numpy as np
import pandas as pd
import os
import math
import bisect
from collections import Counter
import tqdm
 
class YOLO_Kmeans:
 
    def __init__(self, cluster_number, data_file, anchors_file):
        self.cluster_number = cluster_number#6 or 9
        self.data_file = data_file
        self.anchors_file = anchors_file
    
    def init_centroids(self, boxes, n_anchors):
        # k-means++ 初始化，尽量使得质心之间的距离较大
        centroids = []
        boxes_num = len(boxes)
        centroids_index = np.random.choice(boxes_num, 1)[0]
        centroids.append(boxes[centroids_index])
        
        for _ in tqdm.tqdm(range(n_anchors - 1)):
            sum_distance = 0
            distance_list = []
            cur_sum = 0
            
            for box in boxes:
                min_distance = 1
                for centroid_i, centroid in enumerate(centroids):
                    distance = (1 - self.iou(box[np.newaxis], centroid[np.newaxis]))[0][0]
                    if distance < min_distance:
                        min_distance = distance
                sum_distance += min_distance
                distance_list.append(min_distance)
            
            distance_np = np.array(distance_list)
            med_val = np.median(distance_np)
            # med_val = np.mean(distance_np)
            index = np.where(distance_np >= med_val)[0]
            new_centroid_index = np.random.choice(index, 1)[0]
            new_centroid = boxes[new_centroid_index]
            centroids.append(new_centroid)
           
            
        return np.array(centroids)
            
        
    def iou(self, boxes, anchors):
        """
        Calculate the IOU between boxes and anchors.

        :param boxes: 2-d array, shape(n, 2)
        :param anchors: 2-d array, shape(k, 2)
        :return: 2-d array, shape(n, k)
        """
        # Calculate the intersection,
        # the new dimension are added to construct shape (n, 1) and shape (1, k),
        # so we can get (n, k) shape result by numpy broadcast
        w_min = np.minimum(boxes[:, 0, np.newaxis], anchors[np.newaxis, :, 0])
        h_min = np.minimum(boxes[:, 1, np.newaxis], anchors[np.newaxis, :, 1])
        inter = w_min * h_min
           
        # Calculate the union
        box_area = boxes[:, 0] * boxes[:, 1]
        anchor_area = anchors[:, 0] * anchors[:, 1]
        union = box_area[:, np.newaxis] + anchor_area[np.newaxis]

        return inter / (union - inter)
 
   
 
    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy
 
    def kmeans(self, boxes, k, dist=np.median, plus=False):
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        if plus:
            clusters = self.init_centroids(boxes, k)
        else:
            clusters = boxes[np.random.choice(
                box_number, k, replace=False)]  # init k clusters从原来的众多box中随机选取k个box
        
        self.result2txt(clusters, suffix='clusters')
        
        while True: 
            distances = 1 - self.iou(boxes, clusters)
 
            current_nearest = np.argmin(distances, axis=1)
            # print(Counter(current_nearest))
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster], axis=0)
 
            last_nearest = current_nearest
 
        return clusters
 
    def result2txt(self, data, suffix=''):#将data保存进txt文档中，覆盖方式
        f = open(suffix + '_' + self.anchors_file, 'w')
        row = np.shape(data)[0]
        string = ''
        string += '['
        for item in data:
            string = string + '[' + str(item[0]) + ',' + str(item[1]) + '],\n'
        string += ']'
        f.write(string)
        f.close()
 
        
    def txt2boxes(self, is_filter=True):
        data_files = os.listdir(self.data_file)
        dataSet = []
        for file in data_files:
            with open(os.path.join(self.data_file, file), 'r') as f:
                data = f.readlines()
            if len(data) == 0:
                continue
            for item in data:
                item = item.strip().split(' ')
                width = float(item[-2])
                height = float(item[-1])
                dataSet.append([width, height])
        
        if is_filter: #将距离相近的点删除
            new_set = []
            while len(dataSet):
                first_val = dataSet.pop(0)
                new_set.append(first_val)
                for val in dataSet:
                    temp1 = (val[1]-first_val[1]) ** 2 + (val[0]-first_val[0])**2
                    temp2 = first_val[0]**2 + first_val[1]**2
                    res = math.sqrt(temp1 / temp2)
                    if res < 0.2:
                        dataSet.pop(0)
            dataSet = new_set
        result = np.array(dataSet)            
        return result
        
 
    def csv2boxes(self):
        dataSet = []
        train_data = pd.read_csv(self.data_file, header=None)
        for value_i in train_data.values:
            end_num = len(value_i)
            for i in range(1, end_num, 5):
                width = value_i[i+2] - value_i[i]
                height = value_i[i+3] - value_i[i+1]
                dataSet.append([width, height])##获取所有box的宽和高
        result = np.array(dataSet)
        return result
 
    def txt2clusters(self, input_shape=None):
        all_boxes = self.txt2boxes()#获取所有box的宽和高
        result = self.kmeans(all_boxes, k=self.cluster_number, plus=False)
        result = result[np.lexsort(result.T[0, None])]#对于聚类结果，按照第一维度进行从小到大排序
        self.result2txt(result)
        print("K anchors:\n {}".format(result))
        print("Accuracy: {:.2f}%".format(
            self.avg_iou(all_boxes, result) * 100))
        if input_shape is not None:
            result *= input_shape
            print("K reshape anchors:\n {}".format(result))
 
 
if __name__ == "__main__":
    cluster_number = 15  # tiny-yolo--6, yolo--9  
    input_shape = [1280, 720] # [1280, 720] [w, h]
    data_file = "../Annotations-yolo"
    anchors_file = "anchors.txt"    
    kmeans = YOLO_Kmeans(cluster_number=cluster_number, data_file=data_file, anchors_file=anchors_file)
    kmeans.txt2clusters(input_shape=input_shape)
