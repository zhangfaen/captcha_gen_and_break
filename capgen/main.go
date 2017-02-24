// $go run main.go -len 4 -width 80 -height 35 -numOfImages 40  xx/
package main

import (
	"flag"
	"fmt"
	"image"
	// "io"
	"log"
	"os"
	"strings"
	"bytes"

	"github.com/zhangfaen/captcha"
)

var (
	flagImage       = flag.Bool("i", true, "output image captcha")
	flagAudio       = flag.Bool("a", false, "output audio captcha")
	flagLang        = flag.String("lang", "en", "language for audio captcha")
	flagLen         = flag.Int("len", captcha.DefaultLen, "length of captcha")
	flagImgW        = flag.Int("width", captcha.StdWidth, "image captcha width")
	flagImgH        = flag.Int("height", captcha.StdHeight, "image captcha height")
	flagNumOfImages = flag.Int("numOfImages", 1, "how many images will be generated")
)

func usage() {
	fmt.Fprintf(os.Stderr, "usage: captcha [flags] filename\n")
	flag.PrintDefaults()
}

// Get the bi-dimensional pixel array
func getPixels(buf *bytes.Buffer) ([][]Pixel, error) {
	img, _, err := image.Decode(buf)

	if err != nil {
		return nil, err
	}

	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y

	var pixels [][]Pixel
	for y := 0; y < height; y++ {
		var row []Pixel
		for x := 0; x < width; x++ {
			row = append(row, rgbaToPixel(img.At(x, y).RGBA()))
		}
		pixels = append(pixels, row)
	}

	return pixels, nil
}

// img.At(x, y).RGBA() returns four uint32 values; we want a Pixel
func rgbaToPixel(r uint32, g uint32, b uint32, a uint32) Pixel {
	return Pixel{int(r / 257), int(g / 257), int(b / 257), int(a / 257)}
}

// Pixel struct example
type Pixel struct {
	R int
	G int
	B int
	A int
}

func printCSV(pixels [][]Pixel) string {
	ret := ""
	for i := 0; i < len(pixels); i++ {
		for j := 0; j < len(pixels[i]); j++ {
			gray := pixels[i][j].R+pixels[i][j].G+pixels[i][j].B
			placeHolder := ","
			if j == 0 {
				placeHolder = ""
			}
			if gray > 100 {
				ret = ret + placeHolder + fmt.Sprint("1")
			} else {
				ret = ret + placeHolder + "0"
			}

		}
		ret = ret + "\n"
	}
	return ret
}

func main() {
	flag.Parse()
	fname := flag.Arg(0)
	if fname == "" {
		usage()
		os.Exit(1)
	}
	os.MkdirAll(fname, 0777)
	csvFile, _ := os.Create(fname + "data.csv")
	fmt.Fprintln(csvFile, "" + fmt.Sprint(*flagNumOfImages) + "," + fmt.Sprint(*flagImgH) + "," + fmt.Sprint(*flagImgW))
	for i := 0; i < *flagNumOfImages; i++ {
		d := captcha.RandomDigits(*flagLen)
		s := fmt.Sprintln(d)
		s = s[1 : len(s)-2]
		s = strings.Replace(s, " ", "", -1)
		csvLine := s
		w := captcha.NewImage("", d, *flagImgW, *flagImgH)
		var buf *bytes.Buffer = new(bytes.Buffer)
		w.WriteTo(buf)
		if i % 100 == 1 {
			s = fname + s + ".png"
			f, err := os.Create(s)
			if err != nil {
				log.Fatalf("%s", err)
			}
			_, err = w.WriteTo(f)
			if err != nil {
				log.Fatalf("%s", err)
			}
			f.Close()
		}

		pixels, _ := getPixels(buf)
		fmt.Fprintln(csvFile, csvLine)
		fmt.Fprint(csvFile, printCSV((pixels)))
		// fmt.Println(d)
	}
	csvFile.Close()

}
