package main

import (
	"math"
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
	nyaData.MapFloatColumnSlice("[:4]", func(i int, f float64) float64 {
		return math.Log(f)
	})
	nyaData.ClampColumnSlice("[:]", 0, 1)

	nyaData.PrintSummary()
	nyaDatasetRaw := nyaData.ToSequentialDataset("[:]", "[3]", 60)
	guessLength := 10

	nyaDataset := make([]datasets.DataPoint, len(nyaDatasetRaw)-guessLength)
	for i := range nyaDataset {
		output := make([]float64, guessLength)
		for j := 0; j < guessLength; j++ {
			output[j] = nyaDatasetRaw[i+j].Output[0]
		}
		nyaDataset[i] = datasets.DataPoint{Input: nyaDatasetRaw[i].Input, Output: output}
	}

	datasets.SaveDataset(nyaDataset, "data", "NYA_LSTM_Data")
}

func train() {
	nyaData := datasets.OpenDataset("data", "NYA_LSTM_Data")
	trainingData, testingData := nyaData[:11000], nyaData[11000:]

	network := networks.Sequential{}
	network.Initialize(60*5,
		&layers.LSTMLayer{
			Outputs:        20,
			OutputSequence: true,

			InputSize:           15,
			ConstantLengthInput: true,
		},
		&layers.LSTMLayer{
			Outputs:        20,
			OutputSequence: true,

			InputSize:           80,
			ConstantLengthInput: true,
		},
		&layers.LinearLayer{Outputs: 10},
		&layers.ReluLayer{},
	)

	network.BatchSize = 128
	network.SubBatch = 16
	network.LearningRate = 1
	network.Optimizer = &optimizers.AdaGrad{Epsilon: 0.1}

	network.Train(trainingData, testingData, 20*time.Second)

	network.TestOnAndLog(trainingData)
	network.Save("savednetworks", "LSTM_Network")
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
