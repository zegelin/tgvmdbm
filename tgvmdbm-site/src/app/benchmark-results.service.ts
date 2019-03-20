import { Injectable } from '@angular/core';
import { Observable, of } from "rxjs";
import { HttpClient, HttpHeaders } from '@angular/common/http';
import {map, tap} from "rxjs/operators";

@Injectable({
  providedIn: 'root'
})
export class BenchmarkResultsService {
  constructor(private http: HttpClient) { }

  getBenchmarkResults(): Observable<any[]> {
    return this.http.get<any[]>('assets/results.json').pipe(
      map(results => results.slice(0, 10))
    );
  }
}
