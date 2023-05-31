package main

import (
	"fmt"
	"math"
	"os"
	"time"

	"github.com/EganBoschCodes/lossless/datasets"
	"github.com/EganBoschCodes/lossless/neuralnetworks/layers"
	"github.com/EganBoschCodes/lossless/neuralnetworks/networks"
	"github.com/EganBoschCodes/lossless/neuralnetworks/optimizers"
	"github.com/EganBoschCodes/lossless/neuralnetworks/save"
	"github.com/EganBoschCodes/lossless/utils"
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
	nyaDatasetRaw := nyaData.ToSequentialDataset("[:]", "[3]", 240)
	guessLength, guessWindow := 10, 100

	nyaDataset := make([]datasets.DataPoint, len(nyaDatasetRaw)-guessLength*guessWindow)
	for i := range nyaDataset {
		output := make([]float64, guessLength)
		for j := 0; j < guessLength; j++ {
			output[j] = nyaDatasetRaw[i+(j+1)*guessWindow].Output[0]
		}
		nyaDataset[i] = datasets.DataPoint{Input: nyaDatasetRaw[i].Input, Output: output}
	}

	datasets.SaveDataset(nyaDataset, "data", "NYA_LSTM_Data")

	fmt.Printf("Input: %.2f\nOutput: %.2f\n", nyaDataset[0].Input, nyaDataset[0].Output)
}

func train() {
	nyaData := datasets.OpenDataset("data", "NYA_LSTM_Data")
	trainingData, testingData := nyaData[:11000], nyaData[11000:]

	network := networks.Sequential{}
	network.Initialize(240*5,
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
		&layers.LinearLayer{Outputs: 50},
		&layers.LanhLayer{},
		&layers.LinearLayer{Outputs: 10},
		&layers.ReluLayer{},
	)

	network.BatchSize = 128
	network.SubBatch = 16
	network.LearningRate = 1
	network.Optimizer = &optimizers.AdaGrad{Epsilon: 0.1}

	network.Train(trainingData, testingData, 60*time.Second)

	network.TestOnAndLog(trainingData)
	network.Save("savednetworks", "LSTM_Network")
}

func retrain() {
	nyaData := datasets.OpenDataset("data", "NYA_LSTM_Data")
	trainingData, testingData := nyaData[:11000], nyaData[11000:]

	network := networks.Sequential{}
	network.Open("savednetworks", "LSTM_Network")

	network.BatchSize = 128
	network.SubBatch = 16
	network.LearningRate = 0.01
	network.Optimizer = &optimizers.AdaGrad{Epsilon: 0.1}

	network.Train(trainingData, testingData, 60*time.Second)

	network.Save("savednetworks", "LSTM_Network_Retrained")
}

func test() {
	nyaData := datasets.OpenDataset("data", "NYA_LSTM_Data")

	network := networks.Sequential{}
	network.Open("savednetworks", "LSTM_Network")

	network.TestOnAndLog(nyaData)

	csvString := ""
	for _, datapoint := range nyaData {
		output := network.Evaluate(datapoint.Input)
		csvString += fmt.Sprintf("%.6f,%.6f\n", datapoint.Input[len(datapoint.Input)-2], utils.LastOf(output))
	}

	save.WriteToFile("analysis/output.csv", csvString)
}

func main() {
	switch len(os.Args) {
	case 1:
		train()
	case 2:
		if os.Args[1] == "-prep" || os.Args[1] == "-p" {
			prepareData()
		} else if os.Args[1] == "-test" || os.Args[1] == "-t" {
			test()
		} else if os.Args[1] == "-retrain" || os.Args[1] == "-r" {
			retrain()
		} else {
			panic(os.Args[1] + " is not a valid flag (only -prep, -retrain, or -test works)")
		}
	default:
		panic("this file only takes 0 or 1 arguments!")
	}
}
