package main

import (
	"context"
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"sort"
	"strconv"
	"time"

	"github.com/jackc/pgx/v5"

	"main/colfi"
)

type Prediction struct {
	Item       string
	Prediction float64
}

func main() {
	dataset := colfi.NewDataset()
	start := time.Now()
	loadRatings(dataset, "host="+os.Getenv("PGHOST"), 0)
	loadRatings(dataset, "host="+os.Getenv("PGHOST")+" dbname=soothsayer", 0)
	log.Printf("loading ratings took %s\n", time.Since(start))
	config := colfi.SVDConfig{
		NumFactors: 20,
		Verbose:    true,
	}
	m := colfi.NewSVD(dataset, &config)
	start = time.Now()
	m.Fit(20)
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
		dataset.Append(u, i, r)
		j++
	}
}

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
		dataset.Append(record[0], record[1], rating)
	}
	return dataset
}
