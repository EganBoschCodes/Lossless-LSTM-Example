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

	nyaDataset := nyaData.ToSequentialDataset("[:]", "[:4]", 60)
	datasets.SaveDataset(nyaDataset, "data", "NYA_LSTM_Data")
}

func train() {
	nyaData := datasets.OpenDataset("data", "NYA_LSTM_Data")
	trainingData, testingData := nyaData[:10000], nyaData[10000:]

	network := networks.Sequential{}
	network.Initialize(60*5,
		&layers.LSTMLayer{
			Outputs:        12,
			IntervalSize:   20,
			OutputSequence: true,
		},
		&layers.LinearLayer{Outputs: 28},
		&layers.TanhLayer{},
		&layers.LinearLayer{Outputs: 4},
		&layers.ReluLayer{},
	)

	network.BatchSize = 128
	network.SubBatch = 16
	network.LearningRate = 1
	network.Optimizer = &optimizers.AdaGrad{Epsilon: 0.1}

	network.Train(trainingData, testingData, 30*time.Second)
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
