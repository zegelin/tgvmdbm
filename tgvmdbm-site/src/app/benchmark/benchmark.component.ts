import { Component, OnInit, Input } from '@angular/core';

@Component({
  selector: 'app-benchmark',
  templateUrl: './benchmark.component.html',
  styleUrls: ['./benchmark.component.css']
})
export class BenchmarkComponent implements OnInit {
  @Input() benchmark: any;


  constructor() { }

  ngOnInit() {
  }

}
