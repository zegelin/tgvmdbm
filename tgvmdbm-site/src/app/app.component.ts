import { Component } from '@angular/core';
import { BenchmarkResultsService } from "./benchmark-results.service";

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  benchmarks: any[];

  constructor(private benchmarkResultsService: BenchmarkResultsService) {}

  ngOnInit() {
    this.benchmarkResultsService.getBenchmarkResults()
      .subscribe(results => this.benchmarks = results)
  }
}
