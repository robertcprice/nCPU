// Package main is a minimal Go client for the nCPU synthesis API.
//
// Posts a problem-json to POST /synthesize, asks the server to
// transpile the recovered Mog to Go, and (when --run is set) executes
// the emitted Go program on a fresh input via `go run` so the result
// is visible end-to-end.
//
// The expected workflow is:
//
//   # terminal A — start the server
//   python3 ncpu/synthesis_api/server.py
//
//   # terminal B — run this client
//   go run examples/go_client/main.go
//
// Requires the Go toolchain. Prints one row per demo: server-reported
// method, the emitted Go program, and (if --run) the runtime result.

package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"time"
)

// Demo is one of the problems the client posts to the server. The
// `Run` field is the fresh input we execute the emitted Go on (when
// --run is set).
type Demo struct {
	Title       string
	Description string
	Request     map[string]any
	Run         []any
}

var demos = []Demo{
	{
		Title:       "strictly_increasing",
		Description: "1 iff every adjacent pair arr[i] < arr[i-1].",
		Request: map[string]any{
			"name":      "demo_strictly_increasing",
			"signature": "fn demo_strictly_increasing(arr: [i64]) -> i64",
			"lang":      "go",
			"examples": []map[string]any{
				{"inputs": []any{[]int64{1, 2, 3, 4}}, "expected": 1},
				{"inputs": []any{[]int64{-3, -1, 0, 7, 100}}, "expected": 1},
				{"inputs": []any{[]int64{1, 1, 2}}, "expected": 0},
				{"inputs": []any{[]int64{3, 2, 1}}, "expected": 0},
				{"inputs": []any{[]int64{1, 5, 4, 9}}, "expected": 0},
			},
		},
		Run: []any{[]int64{10, 20, 30, 40, 50}},
	},
	{
		Title:       "first_index_of",
		Description: "First i where arr[i] == target, else -1.",
		Request: map[string]any{
			"name":      "demo_first_index_of_0",
			"signature": "fn demo_first_index_of_0(arr: [i64]) -> i64",
			"lang":      "go",
			"examples": []map[string]any{
				{"inputs": []any{[]int64{0, 1, 2, 3}}, "expected": 0},
				{"inputs": []any{[]int64{1, 2, 3}}, "expected": -1},
				{"inputs": []any{[]int64{0}}, "expected": 0},
				{"inputs": []any{[]int64{1, 0, 0}}, "expected": 1},
			},
		},
		Run: []any{[]int64{0, 0, 0, 0, 0}},
	},
	{
		Title:       "count_distinct",
		Description: "Number of distinct values (empty = 0).",
		Request: map[string]any{
			"name":      "demo_count_distinct",
			"signature": "fn demo_count_distinct(arr: [i64]) -> i64",
			"lang":      "go",
			"examples": []map[string]any{
				{"inputs": []any{[]int64{1, 2, 3}}, "expected": 3},
				{"inputs": []any{[]int64{1, 1, 1}}, "expected": 1},
				{"inputs": []any{[]int64{5, 4, 3, 2, 1}}, "expected": 5},
				{"inputs": []any{[]int64{1, 2, 1, 2, 1}}, "expected": 2},
				{"inputs": []any{[]int64{}}, "expected": 0},
			},
		},
		Run: []any{[]int64{1, 2, 3, 4, 5, 6}},
	},
	{
		Title:       "is_anagram",
		Description: "1 iff the two arrays are permutations of each other.",
		Request: map[string]any{
			"name":      "demo_is_anagram",
			"signature": "fn demo_is_anagram(a: [i64], b: [i64]) -> i64",
			"lang":      "go",
			"examples": []map[string]any{
				{"inputs": []any{[]int64{1, 2, 3}, []int64{3, 1, 2}}, "expected": 1},
				{"inputs": []any{[]int64{1, 1, 2, 2}, []int64{2, 1, 2, 1}}, "expected": 1},
				{"inputs": []any{[]int64{1, 2, 3}, []int64{1, 2, 4}}, "expected": 0},
				{"inputs": []any{[]int64{}, []int64{}}, "expected": 1},
			},
		},
		Run: []any{[]int64{10, 20, 30}, []int64{30, 10, 20}},
	},
}

type serverResponse struct {
	Success    bool    `json:"success"`
	Code       string  `json:"code"`
	Method     string  `json:"method"`
	Error      *string `json:"error"`
	RunOutput  *string `json:"run_output"`
	Holdouts   any     `json:"holdouts"`
	Transpiled *struct {
		Python     string `json:"python"`
		Rust       string `json:"rust"`
		Typescript string `json:"typescript"`
		Go         string `json:"go"`
		Java       string `json:"java"`
	} `json:"transpiled"`
}

func main() {
	url := flag.String("url", "http://127.0.0.1:8093", "synthesis API base URL")
	run := flag.Bool("run", false, "execute the emitted Go on the demo's Run input")
	flag.Parse()

	fmt.Printf("nCPU synthesis Go client — %d demos against %s\n", len(demos), *url)
	fmt.Println("=" + strings.Repeat("=", 70))

	nPass := 0
	for i, demo := range demos {
		fmt.Printf("\n### %d. %s\n", i+1, demo.Title)
		fmt.Println()
		fmt.Println(demo.Description)

		body, err := json.Marshal(demo.Request)
		if err != nil {
			fmt.Printf("  ✗ FAILED to marshal: %v\n", err)
			continue
		}
		resp, err := postJSON(*url+"/synthesize", body)
		if err != nil {
			fmt.Printf("  ✗ FAILED to POST: %v\n", err)
			continue
		}

		fmt.Printf("  success: %v\n", resp.Success)
		fmt.Printf("  method:  %s\n", resp.Method)
		if !resp.Success {
			fmt.Printf("  error:   %s\n", deref(resp.Error))
			continue
		}
		nPass++

		// Show the emitted Go (from the transpiled dict if present,
		// otherwise from the raw Mog code).
		var goCode string
		if resp.Transpiled != nil && resp.Transpiled.Go != "" {
			goCode = resp.Transpiled.Go
		} else {
			goCode = resp.Code
		}
		fmt.Println()
		fmt.Println("  emitted Go:")
		for _, line := range strings.Split(goCode, "\n") {
			fmt.Printf("    %s\n", line)
		}

		// --run: write the emitted Go to a temp file and execute it.
		if *run && demo.Run != nil {
			fmt.Println()
			out, err := compileAndRunGo(goCode, demo.Run)
			if err != nil {
				fmt.Printf("  ✗ run failed: %v\n", err)
			} else {
				fmt.Printf("  runtime result: %s\n", out)
			}
		}

		fmt.Println()
		fmt.Println("-" + strings.Repeat("-", 70))
	}

	fmt.Printf("\n  %d/%d demos solved\n", nPass, len(demos))
}

func deref(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}

func postJSON(url string, body []byte) (*serverResponse, error) {
	client := &http.Client{Timeout: 30 * time.Second}
	req, err := http.NewRequest("POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	var sr serverResponse
	if err := json.Unmarshal(raw, &sr); err != nil {
		return nil, fmt.Errorf("invalid JSON: %w\nbody: %s", err, string(raw))
	}
	return &sr, nil
}

func compileAndRunGo(goCode string, inputs []any) (string, error) {
	// Wrap the function in a main that reads JSON from stdin and
	// calls the function, so a Go client doesn't need a hand-written
	// main per demo.
	wrapped := wrapGoMain(goCode, inputs)
	tmpDir, err := os.MkdirTemp("", "ncpu_go_demo_")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(tmpDir)
	path := tmpDir + "/demo.go"
	if err := os.WriteFile(path, []byte(wrapped), 0644); err != nil {
		return "", err
	}
	cmd := exec.Command("go", "run", path)
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &out
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("%v: %s", err, out.String())
	}
	return strings.TrimSpace(out.String()), nil
}

// wrapGoMain produces a self-contained Go file that imports the
// recovered function, builds a main, and calls the function with
// the demo's input, printing the result.
func wrapGoMain(fnCode string, inputs []any) string {
	// Extract the function name from the signature line: "fn NAME("
	// → "func NAME(" in Go.
	fnName := "fn"
	for _, line := range strings.Split(fnCode, "\n") {
		idx := strings.Index(line, "func ")
		if idx < 0 {
			continue
		}
		rest := line[idx+len("func "):]
		paren := strings.Index(rest, "(")
		if paren < 0 {
			continue
		}
		fnName = strings.TrimSpace(rest[:paren])
		break
	}
	// Build a Go file: package main + the recovered function +
	// a main that calls it.
	return fmt.Sprintf(`package main

import "fmt"

%s

func main() {
	fmt.Println(%s(%s))
}
`,
		fnCode,
		fnName,
		multiArraySetup(inputs),
	)
}

// multiArraySetup returns a Go expression that builds the function
// call from `inputs`. For one input, the call is `f(arr)`. For
// multiple, `f(a, b, c, ...)` with each variable declared inline.
func multiArraySetup(inputs []any) string {
	if len(inputs) == 1 {
		return fmt.Sprintf("[]int64%v", inputs[0])
	}
	names := []string{"a", "b", "c", "d", "e"}
	var decls strings.Builder
	var refs strings.Builder
	for i, in := range inputs {
		if i > 0 {
			decls.WriteString("; ")
			refs.WriteString(", ")
		}
		fmt.Fprintf(&decls, "%s := []int64%v", names[i], in)
		refs.WriteString(names[i])
	}
	return decls.String() + " ; " + refs.String()
}
