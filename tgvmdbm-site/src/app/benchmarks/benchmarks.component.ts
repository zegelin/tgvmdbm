import { Component, OnInit, Input } from '@angular/core';

@Component({
  selector: 'app-benchmarks',
  templateUrl: './benchmarks.component.html',
  styleUrls: ['./benchmarks.component.css']
})
export class BenchmarksComponent implements OnInit {
  @Input() benchmarks: any;

  constructor() { }

  ngOnInit() {
  }

}
