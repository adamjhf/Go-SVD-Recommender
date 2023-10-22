package main

import (
	"context"
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"

	"github.com/jackc/pgx/v5"
	"github.com/olekukonko/tablewriter"

	"main/colfi"
)

type Prediction struct {
	Item       string
	Prediction float64
}

func main() {
	u, i, r := loadRatings("host="+os.Getenv("PGHOST"), 10000000)
	trainset, testset, err := colfi.DatasetsFromSlices(u, i, r, 0.2)
	if err != nil {
		log.Fatalf("error loading datasets: %v", err)
	}
	testParams := colfi.GridSearchParams{
		NumEpochs:  []int{20},
		NumFactors: []int{25, 35, 45},
		Reg:        []float64{0.02},
		LR:         []float64{0.01},
		InitStdDev: []float64{0.1},
	}
	results := colfi.GridSearch(trainset, testset, testParams)
	var data [][]string
	for _, r := range results {
		row := []string{strconv.Itoa(r.NumEpochs), strconv.Itoa(r.NumFactors), fmt.Sprintf("%.3f", r.Reg), fmt.Sprintf("%.3f", r.LR), fmt.Sprintf("%.1f", r.InitStdDev), fmt.Sprintf("%.4f", r.Loss), fmt.Sprintf("%v", r.Runtime)}
		data = append(data, row)
	}

	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"NumEpochs", "NumFactors", "Reg", "LR", "InitStdDev", "Loss", "Runtime"})

	for _, v := range data {
		table.Append(v)
	}
	table.Render()
}

func loadRatings(connString string, limit int) ([]string, []string, []float32) {
	ctx := context.Background()
	conn, err := pgx.Connect(ctx, connString)
	if err != nil {
		log.Fatalf("Unable to connect to database: %v\n", err)
	}
	defer conn.Close(ctx)
	queryFilter := ""
	queryFilter += ` INNER JOIN (SELECT film_id
				   FROM ratings
				   GROUP BY film_id
				   HAVING COUNT(*) >= 500) f
	   ON r.film_id = f.film_id`
	if limit > 0 {
		queryFilter += fmt.Sprintf(" LIMIT %d", limit)
	}
	rows, err := conn.Query(ctx, "SELECT r.user_name, r.film_id, r.rating FROM ratings r"+queryFilter)
	if err != nil {
		log.Fatalf("Unable to get ratings: %v\n", err)
	}
	j := 0
	var u, i string
	var r float64
	var us, is []string
	var rs []float32
	for rows.Next() {
		if j%1000000 == 0 {
			log.Printf("loaded %d rows", j)
		}
		rows.Scan(&u, &i, &r)
		us = append(us, u)
		is = append(is, i)
		rs = append(rs, float32(r))
		j++
	}
	return us, is, rs
}

/*func main() {
	dataset := colfi.NewDataset()
	start := time.Now()
	loadRatings(dataset, "host="+os.Getenv("PGHOST"), 100000)
	loadRatings(dataset, "host="+os.Getenv("PGHOST")+" dbname=soothsayer", 0)
	log.Printf("loading ratings took %s\n", time.Since(start))
	config := colfi.SVDConfig{
		NumFactors: 20,
		Verbose:    true,
	}
	m := colfi.NewSVD(dataset, &config)
	start = time.Now()
	m.Fit(5)
	log.Printf("svd took %s\n", time.Since(start))
	user := "freeth"
	var predictions []Prediction
	start = time.Now()
	for film := range dataset.ItemMap {
		predictions = append(predictions, Prediction{film, m.Predict(user, film)})
	}
	log.Printf("predictions took %s\n", time.Since(start))
	sort.Slice(predictions, func(i, j int) bool {
		return predictions[i].Prediction > predictions[j].Prediction
	})
	for _, pred := range predictions[:50] {
		fmt.Printf("%s: %f\n", pred.Item, pred.Prediction)
	}
}

func loadRatings(dataset *colfi.Dataset, connString string, limit int) {
	ctx := context.Background()
	conn, err := pgx.Connect(ctx, connString)
	if err != nil {
		log.Fatalf("Unable to connect to database: %v\n", err)
	}
	defer conn.Close(ctx)
	queryFilter := ""
	if limit > 0 {
		queryFilter += fmt.Sprintf(" LIMIT %d", limit)
	}
	rows, err := conn.Query(ctx, "SELECT user_name, film_id, rating FROM ratings"+queryFilter)
	if err != nil {
		log.Fatalf("Unable to get ratings: %v\n", err)
	}
	j := 0
	var u, i string
	var r float64
	for rows.Next() {
		if j%1000000 == 0 {
			log.Printf("loaded %d rows", j)
		}
		rows.Scan(&u, &i, &r)
		dataset.Append(u, i, float32(r))
		j++
	}
}*/

func loadRatingsFromCSV(fileName string) *colfi.Dataset {
	f, _ := os.Open(fileName)
	defer f.Close()
	r := csv.NewReader(f)
	dataset := colfi.NewDataset()
	for {
		record, err := r.Read()
		if err != nil {
			break
		}
		rating, _ := strconv.ParseFloat(record[2], 32)
		dataset.Append(record[0], record[1], float32(rating))
	}
	return dataset
}
