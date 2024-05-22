import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

class Neuron:
    def __init__(self, n_inputs, bias=0., weights=None):
        self.b = bias
        if weights:
            self.ws = np.array(weights)
        else:
            self.ws = np.random.rand(n_inputs)

    def _f(self, x):  # activation function (here: leaky_relu)
        return max(x * .1, x)

    def __call__(self,
                 xs):  # calculate the neuron's output: multiply the inputs with the weights and sum the values together, add the bias value,
        # then transform the value via an activation function
        return self._f(xs @ self.ws + self.b)

    def print(self):
        print("weigths: ", self.ws, "bias: ", self.b)
    def description(self):
        return "W: "+ str(np.round(self.ws,2))+ "\nB: "+ str(self.b)


class ANN:
    def __init__(self, n_layers=2, layer_size=4, input_size=3, output_size=1):
        if n_layers < 1 or layer_size < 1 or input_size < 1 or output_size < 1:
            return
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.input_size = input_size
        self.output_size = output_size
        layers = []
        first_layer = [Neuron(input_size) for i in range(layer_size)]
        layers.append(first_layer)
        for i in range(n_layers - 1):
            layer = [Neuron(layer_size) for i in range(layer_size)]
            layers.append(layer)
        self.layers = layers
        self.output_layer = [Neuron(layer_size) for i in range(output_size)]

    def predict(self, input):
        values = []
        for j in range(self.layer_size):
            values.append(self.layers[0][j].__call__(input))
        for i in range(1, self.n_layers):
            new_values = []
            for j in range(self.layer_size):
                new_values.append(self.layers[i][j].__call__(values))
            values = new_values;
        return_values = [i.__call__(values) for i in self.output_layer]
        return return_values

    def print(self):
        for i in range(self.n_layers):
            print("layer ", i, ":")
            for j in self.layers[i]:
                j.print()
        print("output layer :")
        for i in self.output_layer:
            i.print()

    def draw(self):
        g = nx.Graph()
        idxs=[]
        idx=0
        pos_x=0
        pos_y=0
        pos={}
        labels={}
        c=3
        for i in range(self.input_size):
                g.add_node(idx)
                pos[idx]=(0,c/(self.input_size+1)*(i+1))
                idxs.append(idx)
                idx+=1
        for i in range(self.n_layers):
            new_idxs=[]
            for j in range(self.layer_size):
                g.add_node(idx)
                pos[idx] = (i+1, c/(self.layer_size+1)*(j+1))
                labels[idx] = self.layers[i][j].description()
                new_idxs.append(idx)
                for c_idx in idxs:
                    g.add_edge(idx,c_idx)
                idx+=1
            idxs=new_idxs
        for i in range(self.output_size):
            g.add_node(idx)
            pos[idx]=(self.n_layers+1,c/(self.output_size+1)*(i+1))
            labels[idx]=self.output_layer[i].description()
            for c_idx in idxs:
                g.add_edge(idx, c_idx)
            idx+=1
        nx.draw_networkx_labels(g,pos,labels,font_size=8,font_color="red")
        nx.draw_networkx_edges(g,pos)
        nx.draw_networkx_nodes(g,pos)
        plt.show()

a = ANN()
print(a.predict([1, 1, 1]))
a.draw()
