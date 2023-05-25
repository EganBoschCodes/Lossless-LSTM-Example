package main

import (
	"os"
	"time"

	"github.com/EganBoschCodes/lossless/datasets"
	"github.com/EganBoschCodes/lossless/neuralnetworks/layers"
	"github.com/EganBoschCodes/lossless/neuralnetworks/networks"
	"github.com/EganBoschCodes/lossless/neuralnetworks/optimizers"
)

func prepareData() {
	allData := datasets.ReadCSV("csvs/StockData.csv", true)
	nyaData := allData.SelectRowsMatching("Index", &datasets.StringEntry{Value: "NYA"})
	nyaData.DeleteColumns("Index", "Date", "Adj Close", "CloseUSD")
	nyaData.ClampColumnSlice("[:]", 0, 1)

	nyaDataset := nyaData.ToLSTMDataset("[:]", "[:4]")
	datasets.SaveDataset(nyaDataset, "data", "NYA_LSTM_Data")
}

func train() {
	nyaData := datasets.OpenDataset("data", "NYA_LSTM_Data")
	trainingData, testingData := nyaData[:10000], nyaData[10000:]

	network := networks.LSTM{}
	network.Initialize(5, 32,
		[]layers.Layer{
			&layers.LinearLayer{Outputs: 32},
		},
		[]layers.Layer{
			&layers.LinearLayer{Outputs: 32},
		},
		[]layers.Layer{
			&layers.LinearLayer{Outputs: 32},
		},
		[]layers.Layer{
			&layers.LinearLayer{Outputs: 32},
		},
		[]layers.Layer{
			&layers.LinearLayer{Outputs: 18},
			&layers.TanhLayer{},
			&layers.LinearLayer{Outputs: 4},
		},
	)

	network.BatchSize = 128
	network.SubBatch = 16
	network.LearningRate = 0.01
	network.Optimizer = &optimizers.AdaGrad{Epsilon: 0.2}

	network.Train(trainingData, testingData, 60, 10*time.Second)
}

func main() {
	switch len(os.Args) {
	case 1:
		train()
	case 2:
		if os.Args[1] == "-prep" || os.Args[1] == "-p" {
			prepareData()
		} else {
			panic(os.Args[1] + " is not a valid flag (only -prep works)")
		}
	default:
		panic("this file only takes 0 or 1 arguments!")
	}
}
